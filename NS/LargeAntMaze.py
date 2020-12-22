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

import gym
import gym_fastsim

from scoop import futures
from termcolor import colored
import BehaviorDescr
import MiscUtils
from Problem import Problem


class LargAntMaze(Problem):
    def __init__(self, bd_type="generic", max_steps=1000, display=False, assets={}):
        """
        """
        super().__init__()
        self.dim_obs=len(self.env.reset())
        self.dim_act=self.env.action_space.shape[0]
        self.display= display
    
        if(display):
            self.env.enable_display()
            print(colored("Warning: you have set display to True, makes sure that you have launched scoop with -n 1", "magenta",attrs=["bold"]))

        self.max_steps=max_steps

        self.bd_extractor=BehaviorDescr.GenericBD(dims=2,num=1)#dims=2 for position, no orientation, num is number of samples (here we take the last point in the trajectory)
        self.dist_thresh=1 #(norm, in pixels) minimum distance that a point x in the population should have to its nearest neighbour in the archive+pop
                               #in order for x to be considerd novel
        

        #self.maze_im=cv2.imread(assets["env_im"]) if len(assets) else None
        self.num_saved=0


        #self._debug_counter=0

    def close(self):
        self.env.close()

    def get_bd_dims(self):
        return self.bd_extractor.get_bd_dims()

    def __call__(self, ag):
        #print("evaluating agent ", ag._idx)

        if hasattr(ag, "eval"):#in case of torch agent
            ag.eval() 


        task_solved=False
        for i in range(self.max_steps):
            if self.display:
                self.env.render()
                time.sleep(0.01)
            
            action=ag(obs)
            action=action.flatten().tolist() if isinstance(action, np.ndarray) else action
            obs, reward, ended, info=self.env.step(action)
            fitness+=reward
            if self.bd_type=="generic":
                behavior_info.append(info["robot_pos"])
            elif self.bd_type=="learned" or self.bd_type=="learned_frozen":
                z=info["robot_pos"][:2]
                #scale to im size
                real_w=self.env.map.get_real_w()
                real_h=self.env.map.get_real_h()
                z[0]=(z[0]/real_w)*behavior_info.shape[1]
                z[1]=(z[1]/real_h)*behavior_info.shape[0]
        
                behavior_info=cv2.circle(behavior_info, (int(z[0]),int(z[1])) , 2, (255,0,0), thickness=-1)
            
            #check if task solved
            dist_to_goal=np.linalg.norm(np.array(info["robot_pos"][:2])-np.array([self.env.goal.get_x(), self.env.goal.get_y()]))
            if dist_to_goal < self.goal_radius:
                task_solved=True
                ended=True
                break#otherwise the robot might move away from the goal
           


            if ended:
                break
     
        #cv2.imwrite(f"/tmp/meta_observation_samples/obs_{ag._idx}.png", behavior_info)
        #self._debug_counter+=1

        bd=None
        if isinstance(self.bd_extractor, BehaviorDescr.GenericBD):
            bd=self.bd_extractor.extract_behavior(np.array(behavior_info).reshape(len(behavior_info), len(behavior_info[0]))) 
        elif self.bd_type=="learned" or self.bd_type=="learned_frozen":
            bd=self.bd_extractor.extract_behavior(behavior_info)
            if task_solved:
                cv2.imwrite("/tmp/solution.png", behavior_info)
        #pdb.set_trace()
        return fitness, bd, task_solved

    def visualise_bds(self,archive, population, quitely=True, save_to=""):
        """

        """
        pass


if __name__=="__main__":
   
    import Agents
    test_scoop=True
    #test_scoop=False

    if test_scoop:
        hm=HardMaze(bd_type="generic",max_steps=2000,display=False)
        num_agents=100
        random_pop=[Agents.Dummy(in_d=5, out_d=2, out_type="list") for i in range(num_agents)]
        
        t1=time.time()
        results=list(futures.map(hm, random_pop))
        t2=time.time()
        print("time==",t2-t1,"secs")#on my machine with 24 cores, I get a factor of about 5x when using all cores instead of just one

        for i in range(len(random_pop)):
            ind=random_pop[i]
            ind._fitness=results[i][0]
            ind._behavior_descr=results[i][1]
            print(i, ind._fitness, ind._behavior_descr)

