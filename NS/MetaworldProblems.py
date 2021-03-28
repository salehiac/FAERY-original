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
import matplotlib.pyplot as plt
import pdb
import sys
import os
import random

import gym
import metaworld
from scoop import futures
from termcolor import colored

import BehaviorDescr
import MiscUtils
from Problem import Problem

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
    def __init__(self, bd_type="type_1", max_steps=-1, display=False, assets={}, ML1_env_name="pick-place-v2", mode="train"):
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
        """
        super().__init__()

        self.ML1_env_name=ML1_env_name
        self.ml1 = metaworld.ML1(self.ML1_env_name) #constructs the benchmark which is an environment. As this is ML1, only the task (i.e. the goal)
                                          #will vary. So ml1.train_classes is going to be of lenght 1
                                          
        self.env = self.ml1.train_classes[self.ML1_env_name]()  
        self.task = random.choice(self.ml1.train_tasks)#changes goal
        self.env.set_task(self.task)  # Set task
 

        self.dim_obs=self.env.observation_space.shape[0]#in the latest versions of metaworld, it is 39
        self.dim_act=self.env.action_space.shape[0]#should be 4 (end-effector position + grasping activation. there is no orientation)
        self.display= display
        
        self.max_steps=self.env.max_path_length

        self.bd_type=bd_type
        if bd_type=="type_0":#position only
            self.bd_extractor=BehaviorDescr.GenericBD(dims=3,num=6)#dims*num dimensional
        elif bd_type=="type_1":#position + gripper effector distances
            self.bd_extractor=BehaviorDescr.GenericBD(dims=4,num=6)#dims*num dimensional
        elif bd_type=="type_2":#position + gripper effector distances + whether gripper is opening or closing
            self.bd_extractor=BehaviorDescr.GenericBD(dims=5,num=6)#dims*num dimensional
        else:
            raise Exception("Unkown bd type")
        self.dist_thresh=1 #see comment in HardMaze.py
        self.num_saved=0

    def get_end_effector_pose(self):

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

    def __call__(self, ag):

        if hasattr(ag, "eval"):#in case of torch agent
            ag.eval() 

        obs=self.env.reset()

        fitness=0
        behavior_hist=[]
        task_solved=False
        for i in range(self.max_steps):
            if self.display:
                self.env.render()
                time.sleep(0.01)

            action=ag(obs)
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

            fitness+=reward
            if info["success"]:
                task_solved=True
                done=True
            if done:
                break
               
        bd=self.bd_extractor.extract_behavior(np.array(behavior_hist).reshape(len(behavior_hist), len(behavior_hist[0]))) 

        return fitness, bd, task_solved

    def visualise_bds(self,archive, population, quitely=True, save_to=""):
        """
        """
        pass


if __name__=="__main__":

    mtw_mt1=MetaWorldMT1(bd_type="type_1", 
            max_steps=-1, 
            display=True, 
            assets={}, 
            ML1_env_name="pick-place-v2",
            mode="train")
    
    import Agents

    dummy_ag=Agents.SmallFC_FW(idx=-1,
            in_d=39,
            out_d=4,
            num_hidden=2,
            hidden_dim=70,
            non_lin="tanh",
            use_bn=False,
            output_normalisation=mtw_mt1.action_normalisation())

    num_experiments=1
    for ii in range(num_experiments):
        fit, beh_desr, is_solved=mtw_mt1(dummy_ag)
        print(f"fitness=={fit}, beh_desr.shape=={beh_desr.shape}, is_solved=={is_solved}")

