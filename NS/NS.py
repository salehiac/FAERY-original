

import time
import sys
import os
from abc import ABC, abstractmethod
import copy
import functools

import numpy as np
import matplotlib.pyplot as plt
import torch
import deap
from deap import tools as deap_tools
from scoop import futures
import yaml
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

    def eval_agents(self, agents):
        xx=self._map(self.problem, agents)
        for ag_i in range(len(agents)):
            ag_i._fitness=xx[ag_i][0]
            ag_i._behavior_descr=xx[ag_i][1]



    def __call__(self, iters, reinit=False):
        """
        iters  int  number of iterations
        """
        if reinit:
            self.archive.reset()

        parents=copy.deepcopy(self._initial_pop)#pop is a member in order to avoid passing copies to workers
        self.eval_agents(parents)
        for it in range(iters):
            offsprings=self.generate_new_agents(parents)#mutations and crossover happen here  <<= deap can be useful here
            self.eval_agents(offsprings)
            pop=parents+offsprings #all of them have _fitness and _behavior_descr now

            #TODO: kd-trees will be needed here at some point
            self.nov_estimator.update(archive=self.archive, pop=pop)
            for ag_i in range(len(pop)):
                pop[ag_i]._nov=self.nov_estimator(ag_i)

            parents=self.selector(individuals=pop, fit_attr="_nov")
            self.archive.update(pop)

    def generate_new_agents(self):
        pass



if __name__=="__main__":

    if len(sys.argv)!=2:
        raise Exception("Usage: ",sys.argv[0], " <yaml_config>")

    with open(sys.argv[1],"r") as fl:
        config=yaml.load(fl,Loader=yaml.FullLoader)

    arch_types={"list_based": Archives.ListArchive}
    arch=arch_types[config["archive"]["type"]](max_size=config["archive"]["max_size"],
            growth_rate=config["archive"]["growth_rate"],
            growth_strategy=config["archive"]["growth_strategy"],
            removal_strategy=config["archive"]["removal_strategy"])


    nov_estimator= NoveltyEstimators.ArchiveBasedNoveltyEstimator(k=config["hyperparams"]["k"]) if config["novelty_estimator"]["type"]=="archive_based" else NoveltyEstimators.LearnedNovelty()

    mutator_type=config["mutator"]["type"]
    if mutator_type=="gaussian":
        mutator_conf=config["mutator"]["gaussian_params"]
        mu, sigma, indpb = mutator_conf
        mutator=functools.partial(deap_tools.mutGaussian,mu=mu, sigma=sigma, indpb=indpb)
    else:
        raise NotImplementedError("mutation type")

    if config["problem"]["name"]=="hardmaze":
        max_episodes=config["problem"]["max_episodes"]
        bd_type=config["problem"]["bd_type"]
        problem=Problems.HardMaze(bd_type=bd_type,max_episodes=max_episodes)
    else:
        raise NotImplementedError("Problem type")

    if config["selector"]["type"]=="elitist":
        selector=functools.partial(deap_tools.selBest,k=config["hyperparams"]["population_size"])
    else:
        raise NotImplementedError("selector")

    ns=NoveltySearch(arch,
            nov_estimator,
            mutator,
            problem,
            initial_pop,
            selector,
            map_type="scoop")

