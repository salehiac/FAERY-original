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

from threading import Thread, Lock

_mutex = Lock()

class HardMaze(Problem):
    def __init__(self, bd_type="generic", max_steps=2000, display=False, assets={}):
        """
        bd_type  str      available options are 
                              - generic   based on spatial trajectory, doesn't take orientation into account
                              - learned   Encoder based, not implemented yet
                              - engineered here, it's a histogram of states 

        assets   list     Either {} or a single list dict with key,val= env_im","absolute path to the maze *pbm". Only used to display behavior descriptors
        """
        super().__init__()
        self.env = gym.make('FastsimSimpleNavigation-v0')
        self.dim_obs=len(self.env.reset())
        self.dim_act=self.env.action_space.shape[0]
        self.display= display
    
        if(display):
            self.env.enable_display()
            print(colored("Warning: you have set display to True, makes sure that you have launched scoop with -n 1", "magenta",attrs=["bold"]))

        self.max_steps=max_steps

        self.bd_type=bd_type
        if bd_type=="generic":
            self.bd_extractor=BehaviorDescr.GenericBD(dims=2,num=1)#dims=2 for position, no orientation, num is number of samples (here we take the last point in the trajectory)
            self.dist_thresh=1 #(norm, in pixels) minimum distance that a point x in the population should have to its nearest neighbour in the archive+pop
                               #in order for x to be considerd novel
        elif bd_type=="learned_frozen":
            self.bd_extractor=BehaviorDescr.FrozenEncoderBased()
            self.dist_thresh=1
        elif bd_type=="engineered":
            raise NotImplementedError("not implemented- engineered bds")
        else:
            raise Exception("Wrong bd type")
        
        self.goal_radius=42# Note that the diameter returned by self.env.goal.get_diam() is 7.0 by default (which would be 21.0 in the 200x200 image). I'm doubling that for faster experiments.

        self.maze_im=cv2.imread(assets["env_im"]) if len(assets) else None
        self.num_saved=0

        self.best_dist_to_goal=10000000

        #self._debug_counter=0

    def close(self):
        self.env.close()

    def get_bd_dims(self):
        return self.bd_extractor.get_bd_dims()

    def __call__(self, ag):
        #print("evaluating agent ", ag._idx)

        if hasattr(ag, "eval"):#in case of torch agent
            ag.eval() 

        obs=self.env.reset()
        fitness=0
        if self.bd_type=="generic":
            behavior_info=[] 
        elif self.bd_type=="learned" or self.bd_type=="learned_frozen":
            behavior_info=self.maze_im.copy()
        elif self.bd_type=="engineered":
            pass

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
           
            #_mutex.acquire()
            #if dist_to_goal< self.best_dist_to_goal:
            #    self.best_dist_to_goal=dist_to_goal
            #    print("************",dist_to_goal)
            #_mutex.release()


            if ended:
                break
     
        cv2.imwrite(f"/tmp/meta_observation_samples/obs_{ag._idx}.png", behavior_info)
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
        currently only for 2d generic ones of size 1, so bds should be [bd_0, ...] with bd_i of length 2
        """
        if quitely and not(len(save_to)):
            raise Exception("quitely=True requires save_to to be an existing directory")
        arch_l=list(archive)
        pop_l=list(population)
        uu=arch_l+pop_l
        z=[x._behavior_descr for x in uu]
        z=np.concatenate(z,0)
        most_novel_individual_in_pop=np.argmax([x._nov for x in population])
        #pdb.set_trace()
        real_w=self.env.map.get_real_w()
        real_h=self.env.map.get_real_h()
        z[:,0]=(z[:,0]/real_w)*self.maze_im.shape[1]
        z[:,1]=(z[:,1]/real_h)*self.maze_im.shape[0]
        
        maze_im=self.maze_im.copy()
        for pt_i in range(z.shape[0]): 
            if pt_i<len(arch_l):#archive individuals
                color=MiscUtils.colors.blue
                thickness=-1
            else:#population individuals
                #pdb.set_trace()
                color=MiscUtils.colors.green
                #thickness=1
                thickness=-1
            maze_im=cv2.circle(maze_im, (int(z[pt_i,0]),int(z[pt_i,1])) , 3, color=color, thickness=thickness)
        
        maze_im=cv2.circle(maze_im,
                (int(z[len(arch_l)+most_novel_individual_in_pop,0]),int(z[len(arch_l)+most_novel_individual_in_pop,1])) , 3, color=MiscUtils.colors.red, thickness=-1)
        
        goal=self.env.map.get_goals()[0]
        
        maze_im=cv2.circle(maze_im, 
                (int(goal.get_y()*self.maze_im.shape[0]/real_h),int(goal.get_x()*self.maze_im.shape[1]/real_w)),
                3, (0,0,0), thickness=-1)
        maze_im=cv2.circle(maze_im, 
                (int(goal.get_y()*self.maze_im.shape[0]/real_h),int(goal.get_x()*self.maze_im.shape[1]/real_w)),
                int(self.goal_radius*self.maze_im.shape[0]/real_h), (0,0,0), thickness=1)


        if not quitely:
            plt.imshow(maze_im)
            plt.show()
        else:
            if len(save_to):
                b,g,r=cv2.split(maze_im)
                maze_im=cv2.merge([r,g,b])
                cv2.imwrite(save_to+"/hardmaze_2d_bd_"+str(self.num_saved)+".png",maze_im)
                self.num_saved+=1


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

