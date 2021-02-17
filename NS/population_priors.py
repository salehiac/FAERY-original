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

import Archives
import NS
import HardMaze
import NoveltyEstimators
import Agents
import MiscUtils





def _mutate_initial_pop(parents, mutator, agent_factory):
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
            num_samples,
            agent_type="feed_forward"):
        """
        Note: unlike many meta algorithms, the improvements of the outer loop are not based on test data, but on meta observations from the inner loop. So the
        test_sampler here is used for the evaluation of the meta algorithm, not for learning.
        pop_sz         int          population size
        off_sz         int          number of offsprings
        G_outer        int          number of generations in the outer (meta) loop
        G_inner        int          number of generations in the inner loop (i.e. num generations for each QD problem)
        train_sampler  functor      any object/functor/function with a __call__ that returns a list of problems from a training distribution of environments
        test_sampler   function     any object/functor/function with a __call__ that returns a list of problems from a test distribution of environments
        num_samples    int          number of environments to 
        agent_type     str          either "feed_forward" or (TODO) "LSTM"
        """

        self.pop_sz=pop_sz
        self.off_sz=off_sz
        self.G_outer=G_outer
        self.G_inner=G_inner
        self.train_sampler=train_sampler
        self.test_sampler=test_sampler
        self.num_samples=num_samples
        self.agent_type=agent_type
        
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
        initial_pop=_mutate_initial_pop(initial_pop,self.mutator, self.make_ag)

        self.pop=initial_pop

        self.inner_selector=functools.partial(MiscUtils.selBest,k=pop_sz,automatic_threshold=False)

        #deap setups
        deap.creator.create("Fitness2d",deap.base.Fitness,weights=(1.0,1.0,))
        deap.creator.create("LightIndividuals",list,fitness=deap.creator.Fitness2d, ind_i=-1)
        
        

    def __call__(self):
        """
        Outer loop of the meta algorithm
        """

        outer_selector=MiscUtils.NSGA2(k=self.pop_sz)
        for outer_g in range(self.G_outer):
            pbs=self.train_sampler(self.num_samples)
            idx_to_individual={x._idx:x for x in self.pop}
            evolution_table=np.zeros([self.pop_sz, self.G_inner])#evolution_table[i,d]=k  means that environment k was solved at depth (inner_generation) d by mutating original agent i
            for pb_i in rangel(len(bps)):#we can't do that in parallel as the QD algos in this for loop already need that parallelism 
                pb=pbs[i]
                nov_estimator= NoveltyEstimators.ArchiveBasedNoveltyEstimator(k=15)
                arch=Archives.ListArchive(max_size=5000,
                        growth_rate=6,
                        growth_strategy="random",
                        removal_strategy="random")
 
                ns=NS.NoveltySearch(archive=arch,
                        nov_estimator=nov_estimator,
                        mutator=self.mutator,
                        problem=pb,
                        selector=self.inner_selector,
                        n_pop=self.pop_sz,
                        n_offspring=self.off_sz,
                        agent_factory=self.make_ag,
                        visualise_bds_flag=1,#log to file
                        map_type="scoop",#or "std"
                        logs_root="/tmp/",
                        compute_parent_child_stats=0,
                        initial_pop=[x for x in self.pop])
                #do NS
                nov_estimator.log_dir=ns.log_dir_path
                ns.save_archive_to_file=False
                _, solutions, depth =ns(iters=self.G_inner,
                        stop_on_reaching_task=True,#this should NEVER be False in this algorithm
                        save_checkpoints=0)#save_checkpoints is not implemented but other functions already do its job

                for sol in solutions:
                    idx_to_individual[sol._root]._useful_evolvability+=1
                    idx_to_individual[sol._root]._adaptation_speed_lst.append(depth)
                    evolution_table[sol._root, depth]=pb_i

            for ind in self.pop:
                ind._mean_adaptation_speed=np.mean(ind._adaptation_speed_lst)

            #now the meta training part
            light_pop=[]
            for i in range(self.pop_sz):
                light_pop.append(deap.creator.LightIndividuals())
                light_pop[-1].fitness.setValues([self.pop[i]._useful_evolvability, self.pop[i]._mean_adaptation_speed])
                light_pop[-1].ind_i=i

            chosen=deap.tools.selNSGA2(light_pop, self.pop_sz, nd="standard")
            chosen_inds=[x.ind_i for x in chosen]

            self.pop=[self.pop[u] for u in chosen_inds]
            
            #reset evolvbility and adaptation stats
            for ind in self.pop:
                ind._useful_evolvability=0
                ind._mean_adaptation_speed=0
                ind._adaptation_speed_lst=[]

if __name__=="__main__":

    TEST_WITH_RANDOM_2D_MAZES=True
    
    if TEST_WITH_RANDOM_2D_MAZES:

        num_samples=5
        train_sampler=functools.partial(HardMaze.sample_mazes,
                G=6, 
                num_samples=num_samples,
                xml_template_path="../environments/env_assets/maze_template.xml",
                tmp_dir="/tmp/",
                from_dataset="/tmp/mazes_6x6_test/",
                random_goals=True)

        test_sampler=None

        MetaQDForSparseRewards(pop_sz=25,
                off_sz=25,
                G_outer=1,
                G_inner=100,
                train_sampler=train_sampler,
                test_sampler=test_sampler,
                num_samples=num_samples,
                agent_type="feed_forward")




