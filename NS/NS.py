

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
from termcolor import colored
import tqdm
#import cv2

import Archives
import NoveltyEstimators
import BehaviorDescr
import Problems
import Agents

class NoveltySearch:

    def __init__(self,
            archive,
            nov_estimator,
            mutator,
            problem,
            initial_pop,
            selector,
            n_offspring,
            agent_factory,
            map_type="scoop"):
        """
        archive           Archive           object implementing the Archive interface
        nov_estimator     NoveltyEstimator  object implementing the NoveltyEstimator interface. 
        problem           Problem           object that provides a __call__ function taking individual_index returning fitness and behavior_descriptors
        initial_pop       list              list[i] should be an agent compatible with problem (mapping problem state observations to actions)
                                            Currently, considered agents should 
                                                - inherit from list (to avoid issues with deap functions that have trouble with numpy slices)
                                                - provide those fields: _fitness, _behavior_descr, _novelty. This is just to facilitate possible interactions with the deap library
        mutator           Mutator
        selector          function
        n_offspring       int           
        agent_factory     function          used to convert mutated list genotypes back to agent types
        map_type          string            different options for sequential/parallel mapping functions. supported values currently are 
                                            "scoop" distributed map from futures.map
                                            "std"   buildin python map
        """
        self.archive=archive
        self.archive.reset()

        self.nov_estimator=nov_estimator
        self.problem=problem
        self._initial_pop=copy.deepcopy(initial_pop)

        self._map=futures.map if map_type=="scoop" else map

        self.mutator=mutator
        self.selector=selector

        self.n_offspring=n_offspring
        self.agent_factory=agent_factory
       
        if n_offspring!=len(initial_pop):
            print(colored("Warning: len(initial_pop)!=n_offspring. This will result in an additional random selection in self.generate_new_agents", "magenta",attrs=["bold"]))

    def eval_agents(self, agents):
        #print("evaluating agents... map type is set to ",self._map)
        tt1=time.time()
        xx=list(self._map(self.problem, agents))
        tt2=time.time()
        elapsed=tt2-tt1
        for ag_i in range(len(agents)):
            ag=agents[ag_i]
            ag._fitness=xx[ag_i][0]
            ag._behavior_descr=xx[ag_i][1]
         
        return elapsed



    def __call__(self, iters, reinit=False):
        """
        iters  int  number of iterations
        """
        if reinit:
            self.archive.reset()

        parents=copy.deepcopy(self._initial_pop)#pop is a member in order to avoid passing copies to workers
        self.eval_agents(parents)

        tqdm_gen = tqdm.trange(iters, desc='', leave=True)
        for it in tqdm_gen:
            offsprings=self.generate_new_agents(parents)#mutations and crossover happen here  <<= deap can be useful here
            self.eval_agents(offsprings)
            pop=parents+offsprings #all of them have _fitness and _behavior_descr now

            self.nov_estimator.update(archive=self.archive, pop=pop)
            novs=self.nov_estimator()#computes novelty of all population
            for ag_i in range(len(pop)):
                pop[ag_i]._nov=novs[ag_i]
                assert pop[ag_i]._nov is not None , "debug that"

            parents=self.selector(individuals=pop, fit_attr="_nov")
            self.archive.update(pop)
            
            tqdm_gen.set_description(f"Generation {it}/{iters}, archive_size=={len(self.archive)}")
            tqdm_gen.refresh()
        
        return parents

    def generate_new_agents(self, parents):
       
        parents_as_list=[x.get_flattened_weights() for x in parents]
        mutated_genotype=[self.mutator(copy.deepcopy(x)) for x in parents_as_list]#deepcopy is because of deap

        ##debug
        #kk=[]
        #for i in range(len(mutated_genotype)):
        #    kk.append(np.array(mutated_genotype[i][0])-np.array(parents_as_list[i]))
       
        mutated_ags=[self.agent_factory() for x in range(self.n_offspring)]
        kept=random.choices(range(len(mutated_genotype)), k=self.n_offspring)
        for i in range(len(kept)):
            mutated_ags[i].set_flattened_weights(mutated_genotype[kept[i]][0])

        return mutated_ags






if __name__=="__main__":

    if len(sys.argv)!=2:
        raise Exception("Usage: ",sys.argv[0], " <yaml_config>")

    ### create ns component from yaml file
    with open(sys.argv[1],"r") as fl:
        config=yaml.load(fl,Loader=yaml.FullLoader)

    # create archive types (if used)
    arch_types={"list_based": Archives.ListArchive}
    arch=arch_types[config["archive"]["type"]](max_size=config["archive"]["max_size"],
            growth_rate=config["archive"]["growth_rate"],
            growth_strategy=config["archive"]["growth_strategy"],
            removal_strategy=config["archive"]["removal_strategy"])

    # create novelty estimators
    nov_estimator= NoveltyEstimators.ArchiveBasedNoveltyEstimator(k=config["hyperparams"]["k"]) if config["novelty_estimator"]["type"]=="archive_based" else NoveltyEstimators.LearnedNovelty()

    # create behavior descriptors
    if config["problem"]["name"]=="hardmaze":
        max_episodes=config["problem"]["max_episodes"]
        bd_type=config["problem"]["bd_type"]
        problem=Problems.HardMaze(bd_type=bd_type,max_episodes=max_episodes)
    else:
        raise NotImplementedError("Problem type")

    #create selector
    if config["selector"]["type"]=="elitist":
        selector=functools.partial(deap_tools.selBest,k=config["hyperparams"]["population_size"])
    else:
        raise NotImplementedError("selector")

    #create population
    in_dims=problem.dim_obs
    out_dims=problem.dim_act
    num_pop=config["hyperparams"]["population_size"]
    if config["population"]["individual_type"]=="simple_fw_fc":
        def make_ag():
            return Agents.SmallFC_FW(in_d=in_dims,
                out_d=out_dims,
                num_hidden=3,
                hidden_dim=10)
    population=[make_ag() for i in range(num_pop)]
    
    
    # create mutator
    mutator_type=config["mutator"]["type"]
    genotype_len=population[0].get_genotype_len()
    if mutator_type=="gaussian_same":
        mutator_conf=config["mutator"]["gaussian_params"]
        mu, sigma, indpb = mutator_conf["mu"], mutator_conf["sigma"], mutator_conf["indpb"]
        mus = [mu]*genotype_len
        sigmas = [sigma]*genotype_len
        mutator=functools.partial(deap_tools.mutGaussian,mu=mus, sigma=sigmas, indpb=indpb)
    else:
        raise NotImplementedError("mutation type")


    #create NS
    map_t="scoop" if config["use_scoop"] else "std"
    ns=NoveltySearch(archive=arch,
            nov_estimator=nov_estimator,
            mutator=mutator,
            problem=problem,
            initial_pop=population,
            selector=selector,
            n_offspring=config["hyperparams"]["offspring_size"],
            agent_factory=make_ag,
            map_type=map_t)

    if 0:
        elapsed_time=ns.eval_agents(population)
        print("agents evaluated in ", elapsed_time, "seconds (map type == ", map_t,")") # on my DLbox machine with 24 cores, I get 12secs with all of them vs 86secs with a single worker
                                                                                        # (for 200 agents) this is consistent with the 5x to 7x acceleration factor I'd seen before

    #do NS
    final_pop=ns(iters=10)

    
