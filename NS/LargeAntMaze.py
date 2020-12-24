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
import sys
import pdb

import gym
import gym_fastsim

from scoop import futures
from termcolor import colored
import BehaviorDescr
import MiscUtils
from Problem import Problem

sys.path.append("..")
from environments.large_ant_maze.ant_maze import AntObstaclesBigEnv

class LargeAntMaze(Problem):
    def __init__(self, bd_type="generic", max_steps=20000, display=False, assets={}):
        """
        """
        super().__init__()
        xml_path=assets["env_xml"]
        self.env=AntObstaclesBigEnv(xml_path=xml_path, max_ts=max_steps)
        self.env.seed(127)#to avoid having a different environment when evaluating agents after optimisation 

        #display=True
        
        self.dim_obs=self.env.observation_space.shape[0]
        self.dim_act=self.env.action_space.shape[0]
        self.display= display
    
        if(display):
            self.env.render()
            print(colored("Warning: you have set display to True, makes sure that you have launched scoop with -n 1", "magenta",attrs=["bold"]))

        self.max_steps=max_steps

        self.bd_extractor=BehaviorDescr.GenericBD(dims=2,num=16)#dims=2 for position, no orientation, num is number of samples. the behavior_descriptor will be dims*num dimensional
        self.dist_thresh=1 
                
        self.num_gens=0

    def close(self):
        self.env.close()

    def get_bd_dims(self):
        return self.bd_extractor.get_bd_dims()

    def __call__(self, ag):
        """
        evaluates the agent
        returns 
            fitness   augments with the number of solved tasks
            bd        behavior descriptor
            solved    boolean, has the agent solved all tasks
        """
        #print("evaluating agent ", ag._idx)

        if hasattr(ag, "eval"):#in case of torch agent
            ag.eval() 

        obs=self.env.reset()

        fitness=0
        behavior_info=[]
        solved_tasks=[0]*len(self.env.goals)
        solved=False
        for step_i in range(self.max_steps):
            if self.display:
                self.env.render()

            action=ag(obs)
            action=action.flatten().tolist() if isinstance(action, np.ndarray) else action
            obs, reward , ended, info=self.env.step(action)
            
            fitness+=reward#rewards given by step() can either be negative or zero

            last_position=np.array([info["x_position"],info["y_position"]])
            behavior_info.append(last_position.reshape(1,2))

            #if info["is_stuck"]:
            #    print("Ant got stuck at step ",step_i)
            #    #pdb.set_trace()

            for t_idx in range(len(self.env.goals)):
                task=self.env.goals[t_idx]
                #print(task.solved_by(last_position))
                prev_status=solved_tasks[t_idx]
                new_status=task.solved_by(last_position)

                if (not prev_status) and new_status:
                    fitness+=1
                    solved_tasks[t_idx]= True
                    #print(colored(f"solved task {task.color} ", "red", attrs=["bold"]))
        
                    
            if all(solved_tasks):
                solved=True
                ended=True
                fitness+=1
                break

            if ended:
                break
        behavior_info=np.concatenate(behavior_info,0)
        bd=self.bd_extractor.extract_behavior(behavior_info).flatten().reshape(1,-1)
        #pdb.set_trace()

        return fitness, bd, solved

    def visualise_bds(self,archive, population, quitely=True, save_to=""):
        """
        for now archive is ignored
        """
        bds=[x._behavior_descr.reshape(self.bd_extractor.num,self.bd_extractor.dims) for x in population]
        novs=[x._nov for x in population]
        sorted_by_nov=np.argsort(novs).tolist()[::-1]#most novel to least novel

        fits=[x._fitness for x in population]
        sorted_by_fitness=np.argsort(fits).tolist()[::-1]#most novel to least novel
        
        to_plot=sorted_by_nov[:3]
        to_plot+=sorted_by_fitness[:3]

        for i in range(len(bds)):
            if i in to_plot:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal', adjustable='box')
                for j in range(len(self.env.goals)):
                    goal_j=self.env.goals[j]
                    plt.plot(goal_j.coords[0],goal_j.coords[1],color=goal_j.color,marker="o",linestyle="",markersize=25)
                plt.plot(bds[i][:,0],bds[i][:,1],"k")
                plt.xlim(-45,45)
                plt.ylim(-45,45)
                #pdb.set_trace()
                if not quitely:
                    plt.show()
                else:
                    plt.savefig(save_to+f"/large_ant_bd_gen_{self.num_gens}_individual_{i}.png")

                plt.close()
        self.num_gens+=1


if __name__=="__main__":
   
    import Agents
    test_scoop=False
    visualize_agent_behavior=True
    #test_scoop=False

    if test_scoop:
        lam=LargeAntMaze(bd_type="generic",
                max_steps=500,#note that the viewer will go up to self.env.frame_skip*max_steps as well... it skips frames
                display=True,
                assets={"env_xml":"/home/achkan/misc_experiments/guidelines_paper/environments/large_ant_maze/xmls/ant_obstaclesbig2.xml"})
        
        num_agents=10
        random_pop=[Agents.Dummy(in_d=lam.dim_obs, out_d=lam.dim_act, out_type="list") for i in range(num_agents)]
        
        t1=time.time()
        results=list(futures.map(lam, random_pop))
        t2=time.time()
        print("time==",t2-t1,"secs")#on my machine with 24 cores, I get a factor of about 5x when using all cores instead of just one

        for i in range(len(random_pop)):
            ind=random_pop[i]
            ind._fitness=results[i][0]
            ind._behavior_descr=results[i][1]
            print(i, ind._fitness, ind._behavior_descr)

    if visualize_agent_behavior:
        import pickle

        pop_path="/home/achkan/misc_experiments/guidelines_log/ant/32d-bd/NS_log_3524/population_gen_54"

        with open(pop_path, "rb") as fl:
            ags_all=pickle.load(fl)

        #novs=[x._nov for x in ags]
        num_to_keep=3
        fits=[x._fitness for x in ags_all]
        kept=np.argsort(fits)[::-1][:num_to_keep]
        ags=[ags_all[i] for i in kept]



        lam=LargeAntMaze(bd_type="generic",
                max_steps=25000,#note that the viewer will go up to self.env.frame_skip*max_steps as well... it skips frames
                display=True,
                assets={"env_xml":"/home/achkan/misc_experiments/guidelines_paper/environments/large_ant_maze/xmls/ant_obstaclesbig2.xml"})

        for ag in ags:
            f_ag,_, s_ag=lam(ag)
            print("final fitness==", f_ag, "solved_all_tasks==",s_ag)
 

