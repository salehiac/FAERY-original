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
import cv2
import pdb

import procgen #should be imported before gym
import gym

from scoop import futures
from termcolor import colored
import BehaviorDescr
import MiscUtils
import Problem

all_envs=list(gym.envs.registry.all())#for some reason pro
available_procgen_envs=[str(x) for x in all_envs if "procgen" in str(x)]
print("available_procgen_envs:\n",available_procgen_envs)
    
    
env = gym.make('procgen:procgen-caveflyer-v0',render_mode="rgb_array", distribution_mode="exploration", use_backgrounds=False)
obs = env.reset()
step = 0
    while 1:
        obs, rew, done, info = env.step(env.action_space.sample())
        print(f"step {step} reward {rew} done {done}")
        step += 1
        break
        if done:
            break



class CaveFlyExploration(Problem):
    def __init__(self, bd_type="generic", max_steps=500, display=False):
        """
        bd_type  str      available options are 
                              - generic   based on spatial trajectory, doesn't take orientation into account
                              - learned   Encoder based, not implemented yet
                              - engineerd here, it's a histogram of states 

        The games's backgrounds are disabled for now.
        """
        super().__init__()
        self.display= display
        render_mode="human" if display else "rgb_array"
        self.env = gym.make('procgen:procgen-caveflyer-v0',render_mode=render_mode, distribution_mode="exploration", use_backgrounds=False, center_agent=True)
        self.dim_obs=len(self.env.reset())
        self.dim_act=1 #cavefly action_space is of type Discrete(15)
    
        if(display):
            print(colored("Warning: you have set display to True, makes sure that you have launched scoop with -n 1", "magenta",attrs=["bold"]))

        self.max_steps=max_steps

        self.bd_type=bd_type
        if bd_type=="generic":
            self.bd_extractor=BehaviorDescr.GenericBD(dims=2,num=1)#dims=2 for position, no orientation, num is number of samples (here we take the last point in the trajectory)
            self.dist_thresh=1 #(norm, in pixels) minimum distance that a point x in the population should have to its nearest neighbour in the archive+pop
                               #in order for x to be considerd novel

        self.num_saved=0

    def close(self):
        self.env.close()

    def __call__(self, ag):
        #print("evaluating agent ", ag._idx)

        if hasattr(ag, "eval"):#in case of torch agent
            ag.eval()

        obs=self.env.reset()
        fitness=0
        behavior_info=[] 
        for i in range(self.max_steps):
            if self.display:
                self.env.render()
                time.sleep(0.01)
            
            action=ag(obs)
            action=action.flatten().tolist() if isinstance(action, np.ndarray) else action
            obs, reward, ended, info=self.env.step(action)
            fitness+=reward
            if self.bd_type!="learned":
                behavior_info.append(info["robot_pos"])
            else:
                behavior_info.append(obs)
            
            #check if task solved
            if np.linalg.norm(np.array(info["robot_pos"][:2])-np.array([self.env.goal.get_x(), self.env.goal.get_y()])) < self.env.goal.get_diam():
                task_solved=True

            if ended:
                break
        
        bd=self.bd_extractor.extract_behavior(np.array(behavior_info).reshape(len(behavior_info), len(behavior_info[0]))) if self.bd_type!="learned" else None

        task_solved=fitness >= 10 # in exploration mode, enemy ships seem invincible, plus reward for destroying enemy ships is 3.0 and there are only two enemy ships.
                                  # so, if the reward is greater than 6.0 it means that the agent has reached the goal
        return fitness, bd, task_solved

    def visualise_bds(self,archive, population, quitely=True, save_to=""):
        """
        currently only for 2d generic ones of size 1, so bds should be [bd_0, ...] with bd_i of length 2
        """
        pass

if __name__=="__main__":
    pass
