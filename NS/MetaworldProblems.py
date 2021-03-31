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

import copy
import time
import numpy as np
import pdb
import sys
import os
import random

import torch
from scoop import futures
from termcolor import colored

"""This is ugly, but necessary because of repeatability issues with metaworld (the PR that allows setting the seed 
hasn't been merged)"""
with open("../common_config/seed_file","r") as fl:
    lns=fl.readlines()
    assert len(lns)==1, "seed_file should only contain a single seed, nothing more"
    seed_=int(lns[0].strip())
    np.random.seed(seed_)
    random.seed(seed_)
    torch.manual_seed(seed_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
import metaworld
import BehaviorDescr
import MiscUtils
from Problem import Problem


def sample_from_ml1_single_task(bd_type="type_1",num_samples=-1,mode="train",task_name="pick-place-v2",tmp_dir="/tmp/meta_test/"):

    ml1=metaworld.ML1(task_name)
    num_possible_tasks=len(ml1.train_tasks) if mode=="train" else len(ml1.test_tasks)
    assert num_samples<=num_possible_tasks, "too many samples required" #is this reasonnable? There is nothing wrong from sampling the same env multiple times
    if num_samples==-1:
        num_samples=num_possible_tasks
    samples=[]
    print("num_samples==",num_samples)
   
    for s_i in range(num_samples):
        mt1_i=MetaWorldMT1(bd_type=bd_type, max_steps=-1, display=False, assets={}, ML1_env_name=task_name, mode=mode, task_id=-1)
        samples.append(mt1_i)

    np.random.shuffle(samples)

    if 1:
        for x in samples:
            print("*********")
            print(x.env.goal, x.env.obj_init_pos, x.env.obj_init_angle)#for pick and place, only obj_init_pos seems to vary

    return samples
    



class MetaWorldMT1(Problem):
    """
    Problem based on meta-world MT1. A sample environment will be paired with a random task,
    and at test time only that task will vary. That is, evaluation does not concern
    inter-task transfer but transfer to previously unseen goals (e.g. goal placement, or cube placement 
    in the bin-picking envs

    Note that all experiments (I've checked them one by one to make sure) use the same robot and gripper
    and the same 4d action space and 39d observation space, so we can use the same behavior descriptors
    everywhere.

    Note that (from https://github.com/rlworkgroup/metaworld/issues/65) 
    "The gripper orientation is fixed in a palm-down configuration to make exploration
    easier, so no info on orientation is needed on the observation. All environments are solvable
    without changing the orientation (see the scripted policies in metaworld.policies for proof)"

    So, the 4d action space just corresponds to positions and gripper action.
    """
    def __init__(self, bd_type="type_1", max_steps=-1, display=False, assets={}, ML1_env_name="pick-place-v2", mode="train", task_id=-1):
        """
        bd_type        The idea is that it fosters exploration related to the task, for now it doesn't matter if it isn't 
                       optimal. Currently those types are available: (noting N the number of samples)
                           - type_0:
                             in R^(3 * N). Only from gripper position, gripper openness/closeness is not taken into account
                             We let the reward take care of supervising the gripper state.
                           - type_1:
                             vector V in R^(4*N). V[:3,i] will be the position of the gripper at the i-th sample, and V[3,i] will
                             be the distance between the two left/right effectors
                           - type_2:
                             vector V in R^(4*N) x C^N where the part in R^4 is the same as type_1, and where C={0,1} indicates 
                             whether the action sent to the gripper is positive or negative (i.e. asking to be open or closed)

        max_steps      this is dicated here by the ML1 tasks, so this argument is ignored and only present for
                       api compatibility.
        display        self explanatory
        assets         ignored, here for api compatibility issues.
        ML1_env_name   str, should be one of the strings in metaworld.ML1.ENV_NAMES
        mode           "train" or "test"
        task_id        int. If -1, randomly samples tasks (goal positions), which is what you usually want for normal training/testing.
                            If task_id!=-1, forces the use of the task with the given task_id. That can be useful for visualisation and debug.
        """
        super().__init__()

        self.ML1_env_name=ML1_env_name
        self.ml1 = metaworld.ML1(self.ML1_env_name) #constructs the benchmark which is an environment. As this is ML1, only the task (e.g. the goal)
                                          #will vary (note that in for example pick and place, the initial configuratino of the object varies, not the goal).
                                          #So ml1.train_classes is going to be of lenght 1
        
        self.mode=mode

       
        #print("TASK_ID==",task_id)
        if self.mode=="train":
            self.env = self.ml1.train_classes[self.ML1_env_name]()  
            self.task_id = np.random.randint(len(self.ml1.train_tasks)) if task_id==-1 else task_id
            self.task=self.ml1.train_tasks[self.task_id]#changes goal
            self.env.set_task(self.task)  # Set task
        if self.mode=="test":
            self.env = self.ml1.test_classes[self.ML1_env_name]()  
            self.task_id = np.random.randint(len(self.ml1.test_tasks)) if task_id==-1 else task_id
            self.task=self.ml1.test_tasks[self.task_id]#changes goal
            self.env.set_task(self.task)  # Set task

        self.env.seed(seed_)
        #self.env.random_init=False #this doesnt' work, it is automatically set to True again in metaworld


        self.dim_obs=self.env.observation_space.shape[0]#in the latest versions of metaworld, it is 39
        self.dim_act=self.env.action_space.shape[0]#should be 4 (end-effector position + grasping activation. there is no orientation)
        self.display= display
        
        self.max_steps=self.env.max_path_length

        self.bd_type=bd_type
        if bd_type=="type_0":#position only
            self.bd_extractor=BehaviorDescr.GenericBD(dims=3,num=2)#dims*num dimensional
        elif bd_type=="type_1":#position + gripper effector distances
            self.bd_extractor=BehaviorDescr.GenericBD(dims=4,num=2)#dims*num dimensional
        elif bd_type=="type_2":#position + gripper effector distances + whether gripper is opening or closing
            self.bd_extractor=BehaviorDescr.GenericBD(dims=5,num=2)#dims*num dimensional
        elif bd_type=="type_3":
            self.bd_extractor=BehaviorDescr.GenericBD(dims=3,num=1)#final position of manipulated object
        else:
            raise Exception("Unkown bd type")
        self.dist_thresh=0.001 #to avoid adding everyone and their mother to the archive (manually fixed but according to the dimensions of behavior space)
        self.num_saved=0

    def get_task_info(self):
        """
        useful for keeping track of which agent sovles what in metaworld
        """
        dct={}
        dct["mode"]=self.mode
        dct["behavior_descriptor_t"]=self.bd_type
        dct["task_id"]=self.task_id
        dct["global_seed"]=seed_ 
        dct["problem constant"]=(self.env.goal, self.env.obj_init_pos,self.env.obj_init_angle)
        dct["name"]=self.ML1_env_name

        return dct

    def set_env_state(self,state):

        self.env.set_env_state(state)

    def get_end_effector_pos(self):

        return self.env.get_endeff_pos()#this is as far as I know the same as obs[:3]

    def get_gripper_openness(self):
        """
        attention, this returns 0.1*obs[3]
        """

        dist=env._get_site_pos('rightEndEffector')-env._get_site_pos('leftEndEffector')
        return np.linalg.norm(dist)#as far as I know this is obs[3]/10 (no idea why they multiply it by 10 in obs[3])


    def action_normalisation(self):
        """
        returns a function that should be used as the last non-linearity in agents (to constrain actions in an expected interval). If identity, just return ""
        """
        return "tanh"#because self.action_space.high is +1, self.action_space.low is -1.

    def close(self):
        self.env.close()

    def get_bd_dims(self):
        return self.bd_extractor.get_bd_dims()
    
    def get_behavior_space_boundaries(self):
        """
        This is used by br-ns type novelty functions. Not important for meta-world for now
        """
        raise NotImplementedError("not implemented. You can easily get that info from self.env.observation_space.high and .low though.")
        return 

    def __call__(self, ag, forced_init_state=None, forced_init_obs=None):
        """
        if forced_init_state is not None, self.env.reset() wont be called and the evaluation will start from forced_init_state.
        """

        if hasattr(ag, "eval"):#in case of torch agent
            ag.eval() 

        if self.display:
            self.env.render()


        with torch.no_grad():#replace with generic context_magager

            if forced_init_state is None:
                obs=self.env.reset()
            else:
                self.set_env_state(forced_init_state)
                obs=self.env._get_obs()
   
            if forced_init_obs is not None:
                obs=forced_init_obs.copy()

            init_state=self.env.get_env_state()
            init_obs=obs.copy()
            #pdb.set_trace()
            first_action=None

            fitness=0
            behavior_hist=[]
            task_solved=False

            seems_stuck=0
            stuck_thresh=1e-5
            prev_eff_pos=self.get_end_effector_pos()

            for it in range(0,self.max_steps):

                #print(colored(f"it=={it},max_steps={self.max_steps}, seems_stuck={seems_stuck}","magenta",attrs=["bold","reverse"]))
                if self.display:
                    self.env.render()
                    time.sleep(0.01)

                action=ag(obs)
                if first_action is None:
                    first_action=action
                action=action.flatten().tolist() if isinstance(action, np.ndarray) else action
                    
                obs, reward, done, info = self.env.step(action)

                if self.bd_type=="type_0":
                    behavior_hist.append(obs[:3])
                elif self.bd_type=="type_1":
                    behavior_hist.append(obs[:4])#obs[3] is 10*get_gripper_openness(), let's go with that for now
                elif self.bd_type=="type_2":
                    closing_command=float(action[3]>0)
                    beh=np.zeros(5)
                    beh[:4]=obs[:4].copy()
                    beh[4]=closing_command
                    behavior_hist.append(beh)
                elif self.bd_type=="type_3":
                    behavior_hist.append(self.env.obj_init_pos)


                diff_effpos=np.linalg.norm(self.get_end_effector_pos()-prev_eff_pos)
                if diff_effpos<stuck_thresh:
                    seems_stuck+=1
                else:
                    seems_stuck=0
                
                prev_eff_pos=self.get_end_effector_pos()
                
                if seems_stuck>80:#don't set that number too low, since if it doesn't move but closes/opens the gripper it doesn't mena it's stuck
                    #print("seems_stuck==",seems_stuck)
                    #print("stuck. ending episode.")
                    task_solved=False
                    done=True
                    reward-=1
                    
                fitness+=reward
                if info["success"]:
                    task_solved=True#if it is stuck but solves the task, it's okay
                    done=True
               
                if done:
                    #print("task done")
                    #print("success==",info["success"])
                    break

     
                   
            bd=self.bd_extractor.extract_behavior(np.array(behavior_hist).reshape(len(behavior_hist), len(behavior_hist[0]))) 
            complete_traj=np.concatenate([x.reshape(1,-1) for x in behavior_hist],0)

        return fitness, bd, task_solved, complete_traj, init_state, init_obs, first_action
    

    def visualise_bds(self,archive, population, quitely=True, save_to=""):
        """
        """
        if not quitely and self.display:
            raise Exception("there are some issues with mujoco windows. If self.display, set quitely to True. Otherwise, set self.display to False")

        fig = plt.figure()
        ax = plt.axes(projection='3d',adjustable="box")
        color_population="red"
        color_archive="blue"
        #color_closed_gripper="green"
        #color_open_gripper="yellow"
        for x in population:
            bd=x._behavior_descr
            bh=x._complete_trajs
            ax.plot3D(bh[:,0], bh[:,1], bh[:,2], "k--")
            ax.scatter3D(bd[:,0], bd[:,1], bd[:,2], c=color_population)
            #ax.set_aspect('equal') #doesn't support the aspect argument....
            
            #ax.scatter3D(bd[:,0], bd[:,1], bd[:,2], todo)
            
        for x in archive:
            bd_a=x._behavior_descr
            bh_a=x._complete_trajs
            ax.plot3D(bh_a[:,0], bh_a[:,1], bh_a[:,2], "m--")
            ax.scatter3D(bd_a[:,0], bd_a[:,1], bd_a[:,2], c=color_archive)
            #ax.set_aspect('equal') #doesn't support the aspect argument....

        if not quitely:
            plt.show()
        else:
            plt.savefig(save_to+f"/gen_{self.num_saved}.png")
            self.num_saved+=1
            plt.close()




if __name__=="__main__":

    TEST_RANDOM_AGENT=False
    TEST_BEST_AG_FROM_SAVED_POPULATION=True
    TEST_SAMPLER=False



    if TEST_BEST_AG_FROM_SAVED_POPULATION:

        #pop_path="/tmp/NS_log_feWA3Gth8o_106120/population_gen_89"
        #pop_path="//tmp//NS_log_feWA3Gth8o_37942/population_gen_26"
        #pop_path="//tmp//NS_log_feWA3Gth8o_39498/population_gen_40"
        #pop_path="/tmp//NS_log_feWA3Gth8o_44434/population_gen_40"
        #pop_path="/tmp//NS_log_TC8FE2XvSR_115088/population_gen_79"
        pop_path="/tmp//NS_log_TC8FE2XvSR_52606/population_gen_20"
        pop_path="/tmp//NS_log_TC8FE2XvSR_54375/population_gen_5"



        
        import Agents
        import pickle

        with open(pop_path,"rb") as fl:
            population=pickle.load(fl)
            
        fits=[x._fitness for x in population]
        novs=[x._nov for x in population]
        solvers=[i for i in range(len(population)) if population[i]._solved_task]
        print("SOLVERS==",solvers)
        best_ag_id=solvers[0]
        #best_ag_id=np.argmax(fits)
        #best_ag_id=np.argmax(novs)
        print("best_agent saved param sum==",population[best_ag_id]._sum_of_model_params)
        print(f"best_ag_id=={best_ag_id},  best ag param sum==",MiscUtils.get_sum_of_model_params(population[best_ag_id]))


        train_or_test=population[best_ag_id]._task_info["mode"]
        tsk_id=population[best_ag_id]._task_info["task_id"]

        print(colored(f"task_info (saved): {population[0]._task_info}","red"))
        
        mtw_mt1=MetaWorldMT1(bd_type=population[best_ag_id]._task_info["behavior_descriptor_t"], 
                max_steps=-1, 
                display=True, 
                assets={}, 
                ML1_env_name=population[best_ag_id]._task_info["name"],
                mode=train_or_test,
                task_id=tsk_id)
        print(colored(f"task_info (created env): {mtw_mt1.get_task_info()}","red"))

        
        
        
        start_from_state=population[best_ag_id]._last_eval_init_state
        start_from_obs=population[best_ag_id]._last_eval_init_obs

        num_experiments=1
        for ii in range(num_experiments):
            print(colored("best agent","green"))
            print("fitness in archive==",fits[best_ag_id])
            fit, beh_desr, is_solved, _ ,_, the_init_obs, _=mtw_mt1(population[best_ag_id],forced_init_state=start_from_state,forced_init_obs=start_from_obs)
            print(f"fitness=={fit}, beh_desr.shape=={beh_desr.shape}, is_solved=={is_solved}")

    if TEST_SAMPLER:

        print("hello")
        sample_envs=sample_from_ml1_single_task(bd_type="type_1",num_samples=-1,mode="train",task_name="pick-place-v2",tmp_dir="/tmp/meta_test/")

