import time
import sys
import os
from abc import ABC, abstractmethod
import copy
import functools
import random
import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch
import deap
from deap import tools as deap_tools
from scoop import futures
import yaml
import argparse
from termcolor import colored
import tqdm
import pickle

import Archives
import NS
import HardMaze
import NoveltyEstimators
import Agents
import MiscUtils





def _mutate_initial_prior_pop(parents, mutator, agent_factory):
    """
    to avoid a population where all agents have init that is too similar
    """
    parents_as_list=[(x._idx, x.get_flattened_weights()) for x in parents]
    parents_to_mutate=range(len(parents_as_list))
    mutated_genotype=[(parents_as_list[i][0], mutator(copy.deepcopy(parents_as_list[i][1]))) for i in parents_to_mutate]#deepcopy is because of deap

    num_s=len(parents_as_list)
    mutated_ags=[agent_factory() for x in range(num_s)]
    for i in range(num_s):
        mutated_ags[i]._parent_idx=-1
        mutated_ags[i]._root=mutated_ags[i]._idx#it is itself the root of an evolutionnary path
        mutated_ags[i].set_flattened_weights(mutated_genotype[i][1][0])
        mutated_ags[i]._created_at_gen=-1#not important here but should be -1 for the first generation anyway
    
    return mutated_ags
    
def _mutate_prior_pop(n_offspring , parents, mutator, agent_factory):
    """
    mutations for the prior (top-level population)
    """
       
    parents_as_list=[(x._idx, x.get_flattened_weights(), x._root) for x in parents]
    parents_to_mutate=random.choices(range(len(parents_as_list)),k=n_offspring)#note that usually n_offspring>=len(parents)
    mutated_genotype=[(parents_as_list[i][0], mutator(copy.deepcopy(parents_as_list[i][1])), parents_as_list[i][2]) for i in parents_to_mutate]#deepcopy is because of deap

    num_s=n_offspring
    mutated_ags=[agent_factory() for x in range(num_s)]
    kept=random.sample(range(len(mutated_genotype)), k=num_s)
    for i in range(len(kept)):
        mutated_ags[i]._parent_idx=-1 #we don't care
        mutated_ags[i]._root=mutated_ags[i]._idx#it is itself the root of an evolutionnary path
        mutated_ags[i].set_flattened_weights(mutated_genotype[kept[i]][1][0])
        mutated_ags[i]._created_at_gen=-1#we don't care
       
    return mutated_ags
    

class MetaQDForSparseRewards:
    """
    Note: is it really a meta algorithm?

    Yes and No. It does:

        - Have the "meta" bayesian property from Levine's talk: there is a prior Theta that is learnt from a dataset, and then therer is a function phi (which here is QD) that maps
          a new dataset and the prior to a new set of weights (a solution).
        - It can be formulated as such: maximise expectation of performance where the expectation is on priors, and the function in the expectation is a function of both the prior and 
          the new dataset

    However... Usually meta algorithms perform the inner loop on a train dataset and the outer loop on a test dataset. However in our case, the outer loop is based on "meta-data" from
    the inner loop.

    So it is still meta but based on "meta" observations about the learning process.
    """

    def __init__(self, 
            pop_sz,
            off_sz,
            G_outer,
            G_inner,
            train_sampler,
            test_sampler,
            num_train_samples,
            num_test_samples,
            agent_type="feed_forward",
            top_level_log_root="/tmp/"):
        """
        Note: unlike many meta algorithms, the improvements of the outer loop are not based on test data, but on meta observations from the inner loop. So the
        test_sampler here is used for the evaluation of the meta algorithm, not for learning.
        pop_sz               int          population size
        off_sz               int          number of offsprings
        G_outer              int          number of generations in the outer (meta) loop
        G_inner              int          number of generations in the inner loop (i.e. num generations for each QD problem)
        train_sampler        functor      any object/functor/function with a __call__ that returns a list of problems from a training distribution of environments
        test_sampler         function     any object/functor/function with a __call__ that returns a list of problems from a test distribution of environments
        num_train_samples    int          number of environments to use at each outer_loop generation for training
        num_test_samples     int          number of environments to use at each outer_loop generation for testing
        agent_type           str          either "feed_forward" or (TODO) "LSTM"
        top_level_log_root   str          where to save the population after each top_level optimisation
        """

        self.pop_sz=pop_sz
        self.off_sz=off_sz
        self.G_outer=G_outer
        self.G_inner=G_inner
        self.train_sampler=train_sampler
        self.test_sampler=test_sampler
        self.num_train_samples=num_train_samples
        self.num_test_samples=num_test_samples
        self.agent_type=agent_type
        
        if os.path.isdir(top_level_log_root):
            dir_path=MiscUtils.create_directory_with_pid(dir_basename=top_level_log_root+"/meta-learning_"+MiscUtils.rand_string()+"_",remove_if_exists=True,no_pid=False)
            print(colored("[NS info] temporary dir for meta-learning was created: "+dir_path, "blue",attrs=[]))
        else:
            raise Exception("tmp_dir doesn't exist")
        
        self.top_level_log=dir_path
    
        
        dummy_sample=train_sampler(num_samples=1)[0]
        dim_obs=dummy_sample.dim_obs
        dim_act=dummy_sample.dim_act
        normalise_with=dummy_sample.action_normalisation()

        if agent_type=="feed_forward":
            def make_ag():
                return Agents.SmallFC_FW(in_d=dim_obs,
                        out_d=dim_act,
                        num_hidden=3,
                        hidden_dim=10,
                        output_normalisation=normalise_with)
        else:
            raise Exception("agent type unknown")

        self.make_ag=make_ag

        self.mutator=functools.partial(deap_tools.mutPolynomialBounded,eta=10, low=-1.0, up=1.0, indpb=0.1)
        initial_pop=[self.make_ag() for i in range(pop_sz)]
        initial_pop=_mutate_initial_prior_pop(initial_pop,self.mutator, self.make_ag)

        self.pop=initial_pop

        self.inner_selector=functools.partial(MiscUtils.selBest,k=pop_sz,automatic_threshold=False)

        #deap setups
        deap.creator.create("Fitness2d",deap.base.Fitness,weights=(1.0,1.0,))
        deap.creator.create("LightIndividuals",list,fitness=deap.creator.Fitness2d, ind_i=-1)


        self.evolution_tables_train=[]
        self.evolution_tables_test=[]
        
        

    def __call__(self):
        """
        Outer loop of the meta algorithm
        """
        disable_testing=False

        for outer_g in range(self.G_outer):
            pbs=self.train_sampler(num_samples=self.num_train_samples)

            tmp_pop=_mutate_prior_pop(self.off_sz, self.pop, self.mutator, self.make_ag)
            
            evolution_table=-1*np.ones([len(tmp_pop), self.G_inner])#evolution_table[i,d]=k  means that environment k was solved at depth (inner_generation) d by 
                                                                    #mutating original agent i


            for pb_i in range(len(pbs)):#we can't do that in parallel as the QD algos in this for loop already need that parallelism 
                self.inner_loop(pbs[pb_i],
                        pb_i,
                        tmp_pop,#should be passed by ref here
                        evolution_table,
                        test_mode=False)
    

            self.evolution_tables_train.append(evolution_table)
            for ind in tmp_pop:
                if len(ind._adaptation_speed_lst):
                    ind._mean_adaptation_speed=np.mean(ind._adaptation_speed_lst)

            #now the meta training part
            light_pop=[]
            for i in range(len(tmp_pop)):
                light_pop.append(deap.creator.LightIndividuals())
                light_pop[-1].fitness.setValues([tmp_pop[i]._useful_evolvability, -1*(tmp_pop[i]._mean_adaptation_speed)])#the -1 factor is because we want to minimise that speed
                light_pop[-1].ind_i=i

            chosen=deap.tools.selNSGA2(light_pop, self.pop_sz, nd="standard")
            chosen_inds=[x.ind_i for x in chosen]

            self.pop=[tmp_pop[u] for u in chosen_inds]
            
            #reset evolvbility and adaptation stats
            for ind in self.pop:
                ind._useful_evolvability=0
                ind._mean_adaptation_speed=0
                ind._adaptation_speed_lst=[]

            with open(self.top_level_log+"/population_prior_"+str(outer_g),"wb") as fl:
                pickle.dump(self.pop, fl)
            np.savez_compressed(self.top_level_log+"/evolution_table_train_"+str(outer_g), self.evolution_tables_train[-1])


            if not disable_testing:
                test_pbs=self.test_sampler(num_samples=self.num_test_samples)
                
                test_evolution_table=-1*np.ones([self.pop_sz, self.G_inner])#evolution_table[i,d]=k  means that environment k was solved at depth (inner_generation) d by 
                                                                            #mutating original agent i

                for pb_i_test in range(len(test_pbs)):#we can't do that in parallel as the QD algos in this for loop already need that parallelism 
                    self.inner_loop(test_pbs[pb_i_test],
                            pb_i_test,
                            self.pop,
                            test_evolution_table,
                            test_mode=True)

                self.evolution_tables_test.append(test_evolution_table)
                np.savez_compressed(self.top_level_log+"/evolution_table_test_"+str(outer_g), self.evolution_tables_test[-1])


    def inner_loop(self, 
            in_problem,
            in_problem_idx,
            population,
            evolution_table_to_update,
            test_mode):
        """
        evolution_table_to_update is just used for visualisation
        test_mode disable updates to the functions that NSGA2 takes in the top-level (meta) loop
        """
      
        #those are population sizes for the QD algorithms, which are different from the top-level one
        population_size=len(population)
        offsprings_size=population_size
        
        
        idx_to_individual={x._idx:x for x in population}
        idx_to_row={population[i]._idx:i for i in range(population_size)}
       
        nov_estimator= NoveltyEstimators.ArchiveBasedNoveltyEstimator(k=15)
        arch=Archives.ListArchive(max_size=5000,
                growth_rate=6,
                growth_strategy="random",
                removal_strategy="random")
 
        ns=NS.NoveltySearch(archive=arch,
                nov_estimator=nov_estimator,
                mutator=self.mutator,
                problem=in_problem,
                selector=self.inner_selector,
                n_pop=population_size,
                n_offspring=offsprings_size,
                agent_factory=self.make_ag,
                visualise_bds_flag=1,#log to file
                map_type="scoop",#or "std"
                logs_root="/tmp/",
                compute_parent_child_stats=0,
                initial_pop=[x for x in population])
        #do NS
        nov_estimator.log_dir=ns.log_dir_path
        ns.save_archive_to_file=False
        _, solutions=ns(iters=self.G_inner,
                stop_on_reaching_task=True,#this should NEVER be False in this algorithm
                save_checkpoints=0)#save_checkpoints is not implemented but other functions already do its job

        if not len(solutions.keys()):
            print(colored("[NS warning] An environement remained unsolved. This can happen, but it should remain very rare.","red",attrs=["bold"]))
            return

        assert len(solutions.keys())==1, "solutions should only contain solutions from a single generation"
        depth=list(solutions.keys())[0]
        for sol in solutions[depth]:
            if not test_mode:
                idx_to_individual[sol._root]._useful_evolvability+=1
                idx_to_individual[sol._root]._adaptation_speed_lst.append(depth)

            evolution_table_to_update[idx_to_row[sol._root], depth]=in_problem_idx


    def show_evolution_table(self, gen, table):

        x_ticks=[f"$\theta_{i}$" for i in range(table[gen].shape[0])]
        y_ticks=[f"$d_{i}$" for i in range(table[gen].shape[1])]

        MiscUtils.plot_matrix_with_textual_values(table[gen],x_ticks,y_ticks)


if __name__=="__main__":

    TEST_WITH_RANDOM_2D_MAZES=True
    
    if TEST_WITH_RANDOM_2D_MAZES:

        train_dataset_path="/home/achkan/datasets/2d_mazes_6x6_dataset_1/mazes_6x6_train"
        test_dataset_path="/home/achkan/datasets/2d_mazes_6x6_dataset_1/mazes_6x6_test"


        num_train_samples=200
        num_test_samples=10

        train_sampler=functools.partial(HardMaze.sample_mazes,
                G=6, 
                xml_template_path="../environments/env_assets/maze_template.xml",
                tmp_dir="/tmp/",
                from_dataset=train_dataset_path,
                random_goals=False)
        
        
        test_sampler=functools.partial(HardMaze.sample_mazes,
                G=6, 
                xml_template_path="../environments/env_assets/maze_template.xml",
                tmp_dir="/tmp/",
                from_dataset=test_dataset_path,
                random_goals=False)

        algo=MetaQDForSparseRewards(pop_sz=25,
                off_sz=25,
                G_outer=100,
                G_inner=200,
                train_sampler=train_sampler,
                test_sampler=test_sampler,
                num_train_samples=num_train_samples,
                num_test_samples=num_test_samples,
                agent_type="feed_forward",
                top_level_log_root="/home/achkan/misc_experiments/generalisation_paper/")

        algo()
        
        pdb.set_trace()


