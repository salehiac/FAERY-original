

import time
import sys
import os
from abc import ABC, abstractmethod
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import deap
from scooop import futures
#import cv2

class NoveltySearch:

    def __init__(self,
            archive,
            nov_estimator,
            mutator,
            problem,
            initial_pop,
            selector=None,
            map_type="scoop"):
        """
        archive           Archive           object implementing the Archive interface
        nov_estimator     NoveltyEstimator  object implementing the NoveltyEstimator interface. 
        problem           Problem           object that provides
                                                - a __call__ function taking individual_index
                                                - a static _population member. Note that this member should be visible to all scoop workers (e.g. the main script should import Problem)
                                                - a set_static_pop function
                                            It expects each individual to have those fields: _fitness, _behavior_descr
        initial_pop       list              list[i] should be an agent compatible with problem (mapping problem state observations to actions)
                                            Currently, considered agents should 
                                                - inherit from list (to avoid issues with deap functions that have trouble with numpy slices)
                                                - provide those fields: _fitness, _behavior_descr, _novelty
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


    def __call__(self, iters, reinit=False):
        """
        iters  int  number of iterations
        """
        if reinit:
            self.archive.reset()
       
        parents=copy.deepcopy(self._initial_pop)#pop is a member in order to avoid passing copies to workers
        for it in range(iters):
            offsprings=self.generate_new_agents(parents)#mutations and crossover happen here  <<= deap can be useful here
            pop=parents+offsprings
            
            self.problem.set_static_pop(pop)#to avoid sending copies of pop to all workers
            self.map(self.problem, range(len(pop)))#sets fitness and behavior fields of individuals
         
            #TODO: kd-trees will be needed here at some point
            self.nov_estimator.update(archive=self.archive, pop_bds=[x._behavior_descr for x in pop])
            self.map(self.nov_estimator, range(len(pop)))

            parents=self.selector(fit_attr="_nov")
            self.archive.update(pop)
            self.archive.manage_size()

    def generate_new_agents(self):
        pass


if __name__=="__main__":
    pass
