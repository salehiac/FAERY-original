import time
import sys
import os
from abc import ABC, abstractmethod
import copy
import functools
import random
#import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch
import deap
from deap import tools as deap_tools
from scoop import futures
import yaml
import argparse
from termcolor import colored
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
        mutated_ags[i]._root=mutated_genotype[kept[i]][2]
        mutated_ags[i].set_flattened_weights(mutated_genotype[kept[i]][1][0])
        mutated_ags[i]._created_at_gen=-1#we don't care
       
    return mutated_ags
  
def _make_2d_maze_ag():
  """
  because scoop only likes top-level functions/objects...
  """
  return Agents.SmallFC_FW(in_d=5,
      out_d=2,
      num_hidden=3,
      hidden_dim=10,
      output_normalisation="")

def ns_instance(in_problem,
    population,
    mutator,
    inner_selector,
    make_ag,
    G_inner):
      
  #those are population sizes for the QD algorithms, which are different from the top-level one
  population_size=len(population)
  offsprings_size=population_size
  
  nov_estimator= NoveltyEstimators.ArchiveBasedNoveltyEstimator(k=15)
  arch=Archives.ListArchive(max_size=5000,
          growth_rate=6,
          growth_strategy="random",
          removal_strategy="random")
 
  ns=NS.NoveltySearch(archive=arch,
          nov_estimator=nov_estimator,
          mutator=mutator,
          problem=in_problem,
          selector=inner_selector,
          n_pop=population_size,
          n_offspring=offsprings_size,
          agent_factory=make_ag,
          visualise_bds_flag=1,#log to file
          map_type="scoop",#or "std"
          logs_root="//scratchbeta/salehia/tmp_NS_distributed//",
          compute_parent_child_stats=0,
          initial_pop=[x for x in population])
  #do NS
  nov_estimator.log_dir=ns.log_dir_path
  ns.disable_tqdm=True
  ns.save_archive_to_file=False
  _, solutions=ns(iters=G_inner,
          stop_on_reaching_task=True,#this should NEVER be False in this algorithm (at least with the current implementation)
          save_checkpoints=0)#save_checkpoints is not implemented but other functions already do its job

  if not len(solutions.keys()):#environment wasn't solved
      return [],-1

  assert len(solutions.keys())==1, "solutions should only contain solutions from a single generation"
  depth=list(solutions.keys())[0]

  roots=[sol._root for sol in solutions[depth]]

  return roots, depth

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
            top_level_log_root="//scratchbeta/salehia/tmp//"):
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
        
        if os.path.isdir(top_level_log_root):
            dir_path=MiscUtils.create_directory_with_pid(dir_basename=top_level_log_root+"/meta-learning_"+MiscUtils.rand_string()+"_",remove_if_exists=True,no_pid=False)
            print(colored("[NS info] temporary dir for meta-learning was created: "+dir_path, "blue",attrs=[]))
        else:
            raise Exception("tmp_dir doesn't exist")
        
        self.top_level_log=dir_path
    
        self.mutator=functools.partial(deap_tools.mutPolynomialBounded,eta=10, low=-1.0, up=1.0, indpb=0.1)

        ###NOTE: if at some point you add the possibility of resuming from an already given pop, don't forget to rest their _useful_evolvability etc to 0/infinity whatever
        initial_pop=[_make_2d_maze_ag() for i in range(pop_sz)]
        initial_pop=_mutate_initial_prior_pop(initial_pop,self.mutator, _make_2d_maze_ag)

        self.pop=initial_pop

        self.inner_selector=functools.partial(MiscUtils.selBest,k=pop_sz*2,automatic_threshold=False)

        #deap setups
        deap.creator.create("Fitness2d",deap.base.Fitness,weights=(1.0,1.0,))
        deap.creator.create("LightIndividuals",list,fitness=deap.creator.Fitness2d, ind_i=-1)


        self.evolution_tables_train=[]
        self.evolution_tables_test=[]
      
      
    def __call__(self):
        """
        Outer loop of the meta algorithm
        
        While each QD problem inside the inner loop is paralellised, the optimisations on the T sampled environments have been performed sequentially.
        Here's an attempt at parallelising that which, just now, is based on hoping that scoop handles future maps inside future maps well
        """


        disable_testing=False

        for outer_g in range(self.G_outer):
            pbs=self.train_sampler(num_samples=self.num_train_samples)

            offsprings=_mutate_prior_pop(self.off_sz, self.pop, self.mutator, _make_2d_maze_ag)

            tmp_pop=self.pop + offsprings #don't change the order of this concatenation
            

            evolution_table=-1*np.ones([len(self.pop), len(pbs)]) #evolution_table[i,j]=k means that agent i solves env j after k mutations
            idx_to_row={self.pop[i]._idx:i for i in range(len(self.pop))}

            metadata=list(futures.map(ns_instance,
                pbs,#in_problem
                [[x for x in tmp_pop] for i in range(len(pbs))],#population
                [self.mutator for i in range(len(pbs))],#mutator
                [self.inner_selector for i in range(len(pbs))],#inner_selector
                [_make_2d_maze_ag for i in range(len(pbs))],#make_ag
                [self.G_inner for i in range(len(pbs))]))#G_inner

            roots_lst=[m[0] for m in metadata]
            depth_lst=[m[1] for m in metadata]
  
            idx_to_individual={x._idx:x for x in tmp_pop}

            for pb_i in range(len(pbs)):
              rt_i=roots_lst[pb_i]
              d_i=depth_lst[pb_i]
              for rt in rt_i:
                idx_to_individual[rt]._useful_evolvability+=1
                idx_to_individual[rt]._adaptation_speed_lst.append(d_i)
                evolution_table[idx_to_row[rt], pb_i]=d_i


            self.evolution_tables_train.append(evolution_table)
            for ind in tmp_pop:
                if len(ind._adaptation_speed_lst):
                    ind._mean_adaptation_speed=np.mean(ind._adaptation_speed_lst)
            
            #display_f0=[x._useful_evolvability for x in tmp_pop]
            #display_f1=[x._mean_adaptation_speed for x in tmp_pop]
            #print("_useful_evolvability TMP_POP \n",display_f0)
            #print("_mean_adaptation_speed TMP_POP\n",display_f1)
            #print("================================================================================")
             
            #now the meta training part
            light_pop=[]
            for i in range(len(tmp_pop)):
                light_pop.append(deap.creator.LightIndividuals())
                light_pop[-1].fitness.setValues([tmp_pop[i]._useful_evolvability, -1*(tmp_pop[i]._mean_adaptation_speed)])#the -1 factor is because we want to minimise that speed
                light_pop[-1].ind_i=i

            chosen=deap.tools.selNSGA2(light_pop, self.pop_sz, nd="standard")
            chosen_inds=[x.ind_i for x in chosen]

            self.pop=[tmp_pop[u] for u in chosen_inds]
            
            #display_f0=[x._useful_evolvability for x in self.pop]
            #display_f1=[x._mean_adaptation_speed for x in self.pop]
            #print("_useful_evolvability POP \n",display_f0)
            #print("_mean_adaptation_speed POP \n",display_f1)
            #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            
            with open(self.top_level_log+"/population_prior_"+str(outer_g),"wb") as fl:
                pickle.dump(self.pop, fl)
            np.savez_compressed(self.top_level_log+"/evolution_table_train_"+str(outer_g), self.evolution_tables_train[-1])
            
            #reset evolvbility and adaptation stats
            for ind in self.pop:
                ind._useful_evolvability=0
                ind._mean_adaptation_speed=float("inf")
                ind._adaptation_speed_lst=[]

           
            if outer_g%10==0 and not disable_testing:
                test_pbs=self.test_sampler(num_samples=self.num_test_samples)
                
                test_evolution_table=-1*np.ones([self.pop_sz, len(test_pbs)])
                idx_to_row_test={self.pop[i]._idx:i for i in range(len(self.pop))}
                
                test_metadata=list(futures.map(ns_instance,
                    test_pbs,#in_problem
                    [[x for x in self.pop] for i in range(len(test_pbs))],#population
                    [self.mutator for i in range(len(test_pbs))],#mutator
                    [self.inner_selector for i in range(len(test_pbs))],#inner_selector
                    [_make_2d_maze_ag for i in range(len(test_pbs))],#make_ag
                    [self.G_inner for i in range(len(test_pbs))]))#G_inner
                
                for pb_t in range(len(test_pbs)):
                  rt_t=test_metadata[pb_t][0]
                  d_t=test_metadata[pb_t][1]
                  for rt in rt_t:
                    test_evolution_table[idx_to_row_test[rt], pb_t]=d_t

                self.evolution_tables_test.append(test_evolution_table)
                np.savez_compressed(self.top_level_log+"/evolution_table_test_"+str(outer_g), self.evolution_tables_test[-1])

    def test_population(self,
        population,
        in_problem):
        """
        used for a posteriori testing after training is done
        """
      
        population_size=len(population)
        offsprings_size=population_size
        
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
                agent_factory=_make_2d_maze_ag,
                visualise_bds_flag=1,#log to file
                map_type="scoop",#or "std"
                logs_root="//scratchbeta/salehia/test_dir_tmp//",
                compute_parent_child_stats=0,
                initial_pop=[x for x in population])
        #do NS
        nov_estimator.log_dir=ns.log_dir_path
        ns.disable_tqdm=True
        ns.save_archive_to_file=False
        _, solutions=ns(iters=self.G_inner,
                stop_on_reaching_task=True,#this should NEVER be False in this algorithm
                save_checkpoints=0)#save_checkpoints is not implemented but other functions already do its job

        
        assert len(solutions.keys())<=1, "solutions should only contain solutions from a single generation"
        if len(solutions.keys()):
          depth=list(solutions.keys())[0]
        else:
          depth=100000

        return depth


if __name__=="__main__":

    TRAIN_WITH_RANDOM_2D_MAZES=True
    TEST_FROM_POP_ONLY=False
    
    if TRAIN_WITH_RANDOM_2D_MAZES:

        train_dataset_path="/home/salehia/datasets/mazes/mazes_8x8_train"
        test_dataset_path="/home/salehia/datasets/mazes/mazes_8x8_test"

        num_train_samples=1
        num_test_samples=1

        train_sampler=functools.partial(HardMaze.sample_mazes,
                G=8, 
                xml_template_path="../environments/env_assets/maze_template.xml",
                tmp_dir="//scratchbeta/salehia/tmp//",
                from_dataset=train_dataset_path,
                random_goals=False)
        
        
        test_sampler=functools.partial(HardMaze.sample_mazes,
                G=8, 
                xml_template_path="../environments/env_assets/maze_template.xml",
                tmp_dir="//scratchbeta/salehia/tmp//",
                from_dataset=test_dataset_path,
                random_goals=False)

        algo=MetaQDForSparseRewards(pop_sz=24,
                off_sz=24,
                G_outer=100,
                G_inner=150,
                train_sampler=train_sampler,
                test_sampler=test_sampler,
                num_train_samples=num_train_samples,
                num_test_samples=num_test_samples,
                top_level_log_root="/scratchbeta/salehia/generalisation_paper/distributed/")

        algo()
    
    elif TEST_FROM_POP_ONLY:

        train_dataset_path="/home/salehia/datasets/mazes/mazes_8x8_train"
        test_dataset_path="/home/salehia/datasets/mazes/mazes_8x8_test"

        num_train_samples=75
        num_test_samples=1

        train_sampler=functools.partial(HardMaze.sample_mazes,
                G=8, 
                xml_template_path="../environments/env_assets/maze_template.xml",
                tmp_dir="//scratchbeta/salehia/tmp//",
                from_dataset=train_dataset_path,
                random_goals=False)
        
        
        test_sampler=functools.partial(HardMaze.sample_mazes,
                G=8, 
                xml_template_path="../environments/env_assets/maze_template.xml",
                tmp_dir="//scratchbeta/salehia/tmp//",
                from_dataset=test_dataset_path,
                random_goals=False)

        algo=MetaQDForSparseRewards(pop_sz=24,
                off_sz=24,
                G_outer=100,
                G_inner=200,
                train_sampler=train_sampler,
                test_sampler=test_sampler,
                num_train_samples=num_train_samples,
                num_test_samples=num_test_samples,
                top_level_log_root="/scratchbeta/salehia/generalisation_paper/")


        population_dir=sys.argv[1]
        pop_fls=os.listdir(population_dir)
        pop_fls=[x for x in pop_fls if "population_prior_" in x]

        test_pbs=test_sampler(num_samples=num_test_samples)

        population_solutions=[]#population_solutions[i,j]==depth at which population i solves problem j
        for i in range(len(pop_fls)):
          pfl=pop_fls[i]
          with open(population_dir+"/"+pfl,"rb") as fl:
            population_to_test=pickle.load(fl)
         
          pb_solutions=[]
          for pb_i in range(len(test_pbs)):
            depth_val=algo.test_population(population_to_test, test_pbs[pb_i])
            pb_solutions.append(depth_val)
          population_solutions.append(pb_solutions)

        population_solutions=np.array(population_solutions)
        np.savez_compressed("/home/salehia/population_solutions",population_solutions)









