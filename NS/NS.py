# Novelty Search Library.
# Copyright (C) 2020 Sorbonne University
# Maintainer: Achkan Salehi (salehi@isir.upmc.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import time
import sys
import os
import copy
import functools
import random
import pdb
import numpy as np
from pympler import asizeof
import gc

import torch
from scoop import futures
import yaml
import argparse
from termcolor import colored


import matplotlib.pyplot as plt
from deap import tools as deap_tools

import Archives
import NoveltyEstimators
import BehaviorDescr
import Agents
import MiscUtils


class NoveltySearch:

    BD_VIS_DISABLE=0
    BD_VIS_TO_FILE=1
    BD_VIS_DISPLAY=2

    def __init__(self,
            archive,
            nov_estimator,
            mutator,
            problem,
            selector,
            n_pop,
            n_offspring,
            agent_factory,
            visualise_bds_flag,
            map_type="scoop",
            logs_root="/tmp/ns_log/",
            compute_parent_child_stats=False,
            initial_pop=[],#make sure they are passed by deepcopy
            problem_sampler=None):
        """
        archive                      Archive           object implementing the Archive interface. Can be None if novelty is LearnedNovelty1d/LearnedNovelty2d
        nov_estimator                NoveltyEstimator  object implementing the NoveltyEstimator interface. 
        problem                      Problem           None, or object that provides 
                                                            - __call__ function taking individual_index returning (fitness, behavior_descriptors, task_solved_or_not)
                                                            - a dist_thresh (that is determined from its bds) which specifies the minimum distance that should separate a point x from
                                                              its nearest neighbour in the archive+pop in order for the point to be considered as novel. It is also used as a threshold on novelty
                                                              when updating the archive.
                                                           - optionally, a visualise_bds function.
                                                        When problem_sampler is not None, problem should be set to None.
        mutator                      Mutator
        selector                     function
        n_pop                        int 
        n_offspring                  int           
        agent_factory                function          used to 1. create intial population and 2. convert mutated list genotypes back to agent types.
                                                       Currently, considered agents should 
                                                           - inherit from list (to avoid issues with deap functions that have trouble with numpy slices)
                                                           - provide those fields: _fitness, _behavior_descr, _novelty, _solved_task, _created_at_gen
                                                       This is just to facilitate interactions with the deap library
        visualise_bds_flag           int               gets a visualisation of the behavior descriptors (assuming the Problem and its descriptors allow it), and either display it or save it to 
                                                       the logs dir.
                                                       possible values are BD_VIS_TO_FILE and BD_VIS_DISPLAY
        map_type                     string            different options for sequential/parallel mapping functions. supported values currently are 
                                                       "scoop" distributed map from futures.map
                                                       "std"   buildin python map
        logs_root                    str               the logs diretory will be created inside logs_root
        compute_parent_child_stats   bool
        initial_pop                  lst               if a prior on the population exists, it should be supplied here. ATTENTION: it should NOT be passed by ref
        problem_sampler              function          problem_sampler(num_samples=n) should return n instances of the problem 
        """
        self.archive=archive
        if archive is not None:
            self.archive.reset()

        self.nov_estimator=nov_estimator
        self.problem=problem

        if problem_sampler is not None:
            assert self.problem is None, "you have provided a problem sampler and a problem. Please decide which you want, and set the other to None."
            self.problem=problem_sampler(num_samples=1)[0]
        
        self.map_type=map_type
        self._map=futures.map if map_type=="scoop" else map
        print(colored("[NS info] Using map_type "+map_type, "green",attrs=["bold"]))

        self.mutator=mutator
        self.selector=selector

        self.n_offspring=n_offspring
        self.agent_factory=agent_factory
       
        self.num_agent_instances=n_pop#this is important for attributing _idx values to future agents
        
        if not len(initial_pop):
            print(colored("[NS info] No initial population prior, initialising from scratch", "magenta",attrs=["bold"]))
            initial_pop=[self.agent_factory(i) for i in range(n_pop)]
            initial_pop=self.generate_new_agents(initial_pop, generation=0)
        else:
            assert len(initial_pop)==n_pop," this shouldn't happen"
            for x in initial_pop:
                x._created_at_gen=0
        

        self._initial_pop=copy.deepcopy(initial_pop)
      
        assert n_offspring>=len(initial_pop) , "n_offspring should be larger or equal to n_pop"

        self.visualise_bds_flag= visualise_bds_flag

        if os.path.isdir(logs_root):
            self.logs_root=logs_root
            self.log_dir_path=MiscUtils.create_directory_with_pid(dir_basename=logs_root+"/NS_log_"+MiscUtils.rand_string()+"_",remove_if_exists=True,no_pid=False)
            print(colored("[NS info] NS log directory was created: "+self.log_dir_path, "green",attrs=["bold"]))
        else:
            raise Exception("Root dir for logs not found. Please ensure that it exists before launching the script.")

        #for problems for which it is relevant (e.g. hardmaze), keep track of individuals that have solved the task
        self.task_solvers={}#key,value=generation, list(agents)

        self.compute_parent_child_stats=compute_parent_child_stats
        
        self.save_archive_to_file=True
        self.disable_tqdm=False


    def eval_agents(self, agents):
        #print("evaluating agents... map type is set to ",self._map)
        tt1=time.time()
        xx=list(self._map(self.problem, agents))#attention, don't deepcopy the problem instance, see Problem.py. Use problem_sampler if necessary instead.
        #xx=list(map(self.problem, agents))
        tt2=time.time()
        elapsed=tt2-tt1
        task_solvers=[]
        for ag_i in range(len(agents)):
            ag=agents[ag_i]
            ag._fitness=xx[ag_i][0]
            ag._behavior_descr=xx[ag_i][1]
            ag._solved_task=xx[ag_i][2]
            ag._complete_trajs=xx[ag_i][3]#for debug only, disable it later
            ag._last_eval_init_state=xx[ag_i][4]
            ag._last_eval_init_obs=xx[ag_i][5]
            ag._first_action=xx[ag_i][6]
                       
            if hasattr(self.problem, "get_task_info"):
                ag._task_info=self.problem.get_task_info()
                
            
            if ag._solved_task:
                task_solvers.append(ag)

            
        return task_solvers, elapsed



    def __call__(self, iters, stop_on_reaching_task=True, reinit=False, save_checkpoints=0):
        """
        iters  int  number of iterations
        """
        with torch.no_grad():
            print(f"Starting NS with pop_sz={len(self._initial_pop)}, offspring_sz={self.n_offspring}")
            print("Evaluation will take time.")

            if save_checkpoints:
                raise NotImplementedError("checkpoint save/load not implemented yet")
            
            if reinit and self.archive is not None:
                self.archive.reset()


            parents=copy.deepcopy(self._initial_pop)#pop is a member in order to avoid passing copies to workers
            self.eval_agents(parents)
       
            self.nov_estimator.update(archive=[], pop=parents)
            novs=self.nov_estimator()#computes novelty of all population
            for ag_i in range(len(parents)):
                parents[ag_i]._nov=novs[ag_i]
                

            #tqdm_gen = tqdm.trange(iters, desc='', leave=True, disable=self.disable_tqdm)
            for it in range(iters):

                print(colored(f"iter=={it}, archive_size={len(self.archive)}","red"))

                offsprings=self.generate_new_agents(parents, generation=it+1)#mutations and crossover happen here  <<= deap can be useful here
                task_solvers, _ =self.eval_agents(offsprings)
               
                pop=parents+offsprings #all of them have _fitness and _behavior_descr now

                for x in pop:
                    if x._age==-1:
                        x._age=it+1-x._created_at_gen
                    else:
                        x._age+=1


                self.nov_estimator.update(archive=self.archive, pop=pop)
                novs=self.nov_estimator()#computes novelty of all population
                for ag_i in range(len(pop)):
                    pop[ag_i]._nov=novs[ag_i]
                
                parents_next=self.selector(individuals=pop, fit_attr="_nov")
                
                #for stdout only
                p_report=[(x._idx, x._fitness, x._nov, x._task_info) for x in parents_next]
                print("@@@@@@@@@@@@@@@@@@@@@@@ FIT",max([x[1] for x in p_report]))
                print("@@@@@@@@@@@@@@@@@@@@@@@ NOV",max([x[2] for x in p_report]))


                if self.compute_parent_child_stats:
                    for x in parents_next:
                        if x._bd_dist_to_parent_bd==-1 and x._created_at_gen>0:#otherwise it has already been computed in a previous generation
                            xp=next((s for s in pop if s._idx== x._parent_idx), None)
                            if xp is None:
                                raise Exception("this shouldn't happen")
                            x._bd_dist_to_parent_bd=self.problem.bd_extractor.distance(x._behavior_descr,xp._behavior_descr)

                parents=parents_next
                if hasattr(self.nov_estimator, "train"):
                    self.nov_estimator.train(parents) 
                if self.archive is not None:
                    self.archive.update(parents, offsprings, thresh=self.problem.dist_thresh, boundaries=[0,600],knn_k=15)
                    if self.save_archive_to_file:
                        self.archive.dump(self.log_dir_path+f"/archive_{it}")
                
                #self.visualise_bds(parents + [x for x in offsprings if x._solved_task])
                for ag in parents:
                    ag._sum_of_model_params=MiscUtils.get_sum_of_model_params(ag)

                #for ag in self.archive:
                #    if ag not in parents:
                #        ag.mds=None#to save RAM

                print(colored(f"ARCHIVE SIZE=={asizeof.asizeof(self.archive)/1e6}","magenta"))
                MiscUtils.dump_pickle(self.log_dir_path+f"/population_gen_{it}",parents) ############## TODO: it's a bad idea to save AFTER training... This reduces inital novelty, which is therefore
                                                                                         ############## favoring archive-based methods in comparisons... Hack: for now, I'll take that into account
                                                                                         ############## in comparisons, but this is not clean at all, so yeah, change that  (update: what? did I ever
                                                                                         ############## do any of that?)
                
                if len(task_solvers):
                    print(colored("[NS info] found task solvers (generation "+str(it)+")","magenta",attrs=["bold"]))
                    self.task_solvers[it]=task_solvers
                    if stop_on_reaching_task:
                        break
                gc.collect()

                #tqdm_gen.set_description(f"Generation {it}/{iters}, archive_size=={len(self.archive) if self.archive is not None else -1}")
                #tqdm_gen.refresh()
       
            return parents, self.task_solvers#iteration:list_of_agents


    def generate_new_agents(self, parents, generation:int):
       
        parents_as_list=[(x._idx, x.get_flattened_weights(), x._root) for x in parents]
        parents_to_mutate=random.choices(range(len(parents_as_list)),k=self.n_offspring)#note that usually n_offspring>=len(parents)
        mutated_genotype=[(parents_as_list[i][0], self.mutator(copy.deepcopy(parents_as_list[i][1])), parents_as_list[i][2]) for i in parents_to_mutate]#deepcopy is because of deap

        num_s=self.n_offspring if generation!=0 else len(parents_as_list)
        
        mutated_ags=[self.agent_factory(self.num_agent_instances+x) for x in range(num_s)]
        kept=random.sample(range(len(mutated_genotype)), k=num_s)
        for i in range(len(kept)):
            mutated_ags[i]._parent_idx=mutated_genotype[kept[i]][0]
            mutated_ags[i].set_flattened_weights(mutated_genotype[kept[i]][1][0])
            mutated_ags[i]._created_at_gen=generation
            mutated_ags[i]._root=mutated_genotype[kept[i]][2]

        self.num_agent_instances+=len(mutated_ags)

        for x in mutated_ags:
            x.eval()
       
        return mutated_ags
    
    def visualise_bds(self, agents):
         
        if self.visualise_bds_flag!=NoveltySearch.BD_VIS_DISABLE:# and it%10==0:
            q_flag=True if self.visualise_bds_flag==NoveltySearch.BD_VIS_TO_FILE else False
            archive_it=iter(self.archive) if self.archive is not None else []
            self.problem.visualise_bds(archive_it, agents, quitely=q_flag, save_to=self.log_dir_path )


if __name__=="__main__":


    #please don't abuse the parser. Algorithmic params should be set in the yaml files
    parser = argparse.ArgumentParser(description='Novelty Search.')
    parser.add_argument('--config', type=str,  help="yaml config file for ns", default="")
    parser.add_argument('--resume', type=str,  help="(unused, you can use other parts of the code) resume exectution from a checkpoint directory", default="")

    args = parser.parse_args()

    if len(args.config) and len(args.resume):
        raise Exception("only one of --config or --resume should be provided.")

    if not (len(args.config) or len(args.resume)):
        raise Exception("Either --config or --resume should be provided.")

    if len(args.config):
        ### create ns component from yaml file
        with open(args.config,"r") as fl:
            config=yaml.load(fl,Loader=yaml.FullLoader)
           
        # create behavior descriptors and problem
        if config["problem"]["name"]=="hardmaze":
            max_steps=config["problem"]["max_steps"]
            bd_type=config["problem"]["bd_type"]
            assets=config["problem"]["assets"]
            import HardMaze
            problem=HardMaze.HardMaze(bd_type=bd_type,max_steps=max_steps, assets=assets)
        elif config["problem"]["name"]=="large_ant_maze" or config["problem"]["name"]=="huge_ant_maze":
            max_steps=config["problem"]["max_steps"]
            bd_type=config["problem"]["bd_type"]
            assets=config["problem"]["assets"]
            pb_type="huge" if config["problem"]["name"]=="huge_ant_maze" else "large"
            import LargeAntMaze
            problem=LargeAntMaze.LargeAntMaze(pb_type=pb_type,bd_type=bd_type, max_steps=max_steps, assets=assets)
        elif config["problem"]["name"]=="archimedean_spiral":
            import ArchimedeanSpiral
            problem=ArchimedeanSpiral.ArchimedeanSpiralProblem()
        elif config["problem"]["name"]=="metaworld-ml1":
            import MetaworldProblems
            problem=MetaworldProblems.MetaWorldMT1(bd_type=config["problem"]["bd_type"],
                    max_steps=-1, 
                    display=False,#don't set it to True with scoop
                    assets={}, 
                    ML1_env_name=config["problem"]["env_conf"][0],
                    mode=config["problem"]["env_conf"][1])
        else:
            raise NotImplementedError("Problem type")

        # create novelty estimators and if necessary, archives
        if config["novelty_estimator"]["type"]=="archive_based":
            nov_estimator= NoveltyEstimators.ArchiveBasedNoveltyEstimator(k=config["hyperparams"]["k"])
            arch_types={"list_based": Archives.ListArchive}
            arch=arch_types[config["archive"]["type"]](max_size=config["archive"]["max_size"],
                    growth_rate=config["archive"]["growth_rate"],
                    growth_strategy=config["archive"]["growth_strategy"],
                    removal_strategy=config["archive"]["removal_strategy"])
        elif config["novelty_estimator"]["type"]=="learned":
            bd_dims=problem.get_bd_dims()
            #embedding_dims=bd_dims
            #embedding_dims=1
            embedding_dims=2*bd_dims
            nov_estimator=NoveltyEstimators.LearnedNovelty1d(in_dim=bd_dims, emb_dim=embedding_dims, pb_limits=problem.get_behavior_space_boundaries())
            arch=None

        #create selector
        if config["selector"]["type"]=="elitist_with_thresh":

            selector=functools.partial(MiscUtils.selBest,k=config["hyperparams"]["population_size"])

        elif config["selector"]["type"]=="roulette_with_thresh":
            roulette_msg="Usage currently not supported: it ends up chosing the same element many times, this duplicates agent._ids etc"
            roulette_msg+=" fixing this bug is not a priority since selBest with thresholding actually works well"
            raise Exception(roulette_msg)
            #selector=functools.partial(MiscUtils.selRoulette,k=config["hyperparams"]["population_size"])

        elif config["selector"]["type"]=="nsga2_with_thresh":

            selector=MiscUtils.NSGA2(k=config["hyperparams"]["population_size"])
        
        elif config["selector"]["type"]=="nsga2":
            
            selector=MiscUtils.NSGA2(k=config["hyperparams"]["population_size"])
        
        elif config["selector"]["type"]=="elitist":

            selector=functools.partial(MiscUtils.selBest,k=config["hyperparams"]["population_size"],automatic_threshold=False)

        else:
            raise NotImplementedError("selector")

        #create population
        in_dims=problem.dim_obs
        out_dims=problem.dim_act
        num_pop=config["hyperparams"]["population_size"]
        if config["population"]["individual_type"]=="simple_fw_fc":
            
            normalise_output_with="" 
            num_hidden=3
            hidden_dim=10
            if "large_ant_maze"==config["problem"]["name"]:
                normalise_output_with="tanh"
                num_hidden=4
                hidden_dim=10
            elif "metaworld-ml1"==config["problem"]["name"]:
                normalise_output_with="tanh"
                num_hidden=2#SmallFC_FW actually adds another layer
                hidden_dim=50


            def make_ag(idx):
                return Agents.SmallFC_FW(
                        idx,
                        in_d=in_dims,
                        out_d=out_dims,
                        num_hidden=num_hidden,
                        hidden_dim=hidden_dim,
                        output_normalisation=normalise_output_with)
        elif config["population"]["individual_type"]=="agent1d":
            def make_ag(idx):
                return Agents.Agent1d(idx,min(problem.env.phi_vals), max(problem.env.phi_vals))


        
        # create mutator
        mutator_type=config["mutator"]["type"]
        genotype_len=make_ag(-1).get_genotype_len()
        if mutator_type=="gaussian_same":
            mutator_conf=config["mutator"]["gaussian_params"]
            mu, sigma, indpb = mutator_conf["mu"], mutator_conf["sigma"], mutator_conf["indpb"]
            mus = [mu]*genotype_len
            sigmas = [sigma]*genotype_len
            mutator=functools.partial(deap_tools.mutGaussian,mu=mus, sigma=sigmas, indpb=indpb)

        elif mutator_type=="poly_same":
            mutator_conf=config["mutator"]["poly_params"]
            eta, low, up, indpb = mutator_conf["eta"], mutator_conf["low"], mutator_conf["up"], mutator_conf["indpb"] 

            if config["population"]["individual_type"]=="agent1d":
                dummy_ag=make_ag(-1)
                low=dummy_ag.min_val
                up=dummy_ag.max_val

            mutator=functools.partial(deap_tools.mutPolynomialBounded,eta=eta, low=low, up=up, indpb=indpb)

        else:
            raise NotImplementedError("mutation type")


        #create NS
        map_t="scoop" if config["use_scoop"] else "std"
        visualise_bds=config["visualise_bds"]

        ns=NoveltySearch(archive=arch,
                nov_estimator=nov_estimator,
                mutator=mutator,
                problem=problem,
                selector=selector,
                n_pop=num_pop,
                n_offspring=config["hyperparams"]["offspring_size"],
                agent_factory=make_ag,
                visualise_bds_flag=visualise_bds,
                map_type=map_t,
                logs_root=config["ns_log_root"],
                compute_parent_child_stats=config["compute_parent_child_stats"],
                initial_pop=[],
                problem_sampler=None)

        if 0:
            elapsed_time=ns.eval_agents(population)
            print("agents evaluated in ", elapsed_time, "seconds (map type == ", map_t,")") # on my DLbox machine with 24 cores, I get 12secs with all of them vs 86secs with a single worker
                                                                                            # (for 200 agents) this is consistent with the 5x to 7x acceleration factor I'd seen before

        MiscUtils.bash_command(["cp", args.config, ns.log_dir_path+"/config.yaml"])
        
        #do NS
        stop_on_reaching_task=config["stop_when_task_solved"]
        nov_estimator.log_dir=ns.log_dir_path
        ns.disable_tqdm=config["disable_tqdm"]
        ns.save_archive_to_file=config["archive"]["save_to_file"]
        if ns.disable_tqdm:
            print(colored("[NS info] tqdm is disabled.", "magenta", attrs=["bold"]))
        
        final_pop, solutions=ns(iters=config["hyperparams"]["num_generations"],stop_on_reaching_task=stop_on_reaching_task, save_checkpoints=config["save_checkpoints"])
    
    elif len(args.resume):

        raise Exception("not implemented yet. Technically, it isn't necessary, you can resume just by setting initial_pop in NoveltySearch.__init__")
        


