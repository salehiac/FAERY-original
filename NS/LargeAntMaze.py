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
    def __init__(self, pb_type="huge", bd_type="generic", max_steps=20000, display=False, assets={}):
        """
        pb_type  str   either "huge" or "large"
        """
        super().__init__()
        xml_path=assets["huge_ant_maze"] if pb_type=="huge" else assets["large_ant_maze"]
        self.env=AntObstaclesBigEnv(xml_path=xml_path, max_ts=max_steps, goal_type=pb_type)
        self.env.seed(127)#to avoid having a different environment when evaluating agents after optimisation 

        #display=True
        
        self.dim_obs=self.env.observation_space.shape[0]
        self.dim_act=self.env.action_space.shape[0]
        self.display= display
    
        if(display):
            self.env.render()
            print(colored("Warning: you have set display to True, makes sure that you have launched scoop with -n 1", "magenta",attrs=["bold"]))

        self.max_steps=max_steps

        num_samples=48 if pb_type=="huge" else 16
        self.bd_extractor=BehaviorDescr.GenericBD(dims=2,num=num_samples)#dims=2 for position, the behavior_descriptor will be dims*num dimensional
        self.dist_thresh=1 
                
        self.num_gens=0
    
    def action_normalisation(self):
        """
        returns a function that should be used as the last non-linearity in agents (to constrain actions in an expected interval). If identity, just return ""
        """
        return "tanh"

    def close(self):
        self.env.close()

    def get_bd_dims(self):
        return self.bd_extractor.get_bd_dims()
    
    def get_behavior_space_boundaries(self):
        lam_limits=np.array([[-50,50]])
        lam_limits=np.repeat(lam_limits, self.get_bd_dims(), axis=0)
        return lam_limits


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

        return fitness, bd, solved, None

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
    visualize_agent_behavior=False
    visualise_cover_2d=True
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

        #pop_path="/home/achkan/misc_experiments/guidelines_log/ant/32d-bd/NS_log_11048/population_gen_124"
        pop_path="/home/achkan/misc_experiments/guidelines_log/ant/96d-bd/NS_log_89973/population_gen_800"

        with open(pop_path, "rb") as fl:
            ags_all=pickle.load(fl)

        num_to_keep=3
       
        
        fits=[x._fitness for x in ags_all]
        kept=np.argsort(fits)[::-1][:num_to_keep] 
        print("agents were chosen based on fitness")
        
        #novs=[x._nov for x in ags_all]
        #kept=np.argsort(novs)[::-1][:num_to_keep]
        #print("agents were chosen based on novelty")



        ags=[ags_all[i] for i in kept]



        lam=LargeAntMaze(bd_type="generic",
                pb_type="huge",
                max_steps=25000,#note that the viewer will go up to self.env.frame_skip*max_steps as well... it skips frames
                display=False,
                assets={"large_ant_maze":"/home/achkan/misc_experiments/guidelines_paper/environments/large_ant_maze/xmls/ant_obstaclesbig2.xml", "huge_ant_maze":"/home/achkan/misc_experiments/guidelines_paper/environments/large_ant_maze/xmls/ant_obstacles_huge.xml"})

        if 0:
            for ag in ags:
                f_ag,_, s_ag=lam(ag)
                print("final fitness==", f_ag, "solved_all_tasks==",s_ag)


    if visualise_cover_2d:

        import pickle
        
        dummy=LargeAntMaze(bd_type="generic",
                pb_type="large",
                max_steps=25000,#note that the viewer will go up to self.env.frame_skip*max_steps as well... it skips frames
                display=False,
                assets={"large_ant_maze":"/home/achkan/misc_experiments/guidelines_paper/environments/large_ant_maze/xmls/ant_obstaclesbig2.xml", "huge_ant_maze":"/home/achkan/misc_experiments/guidelines_paper/environments/large_ant_maze/xmls/ant_obstacles_huge.xml"})


        
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.set_aspect('equal', adjustable='box')
        #for j in range(len(dummy.env.goals)):
        #    goal_j=dummy.env.goals[j]
        #    plt.plot(goal_j.coords[0],goal_j.coords[1],color=goal_j.color,marker="o",linestyle="",markersize=25)



        
        #root_dir="/home/achkan/misc_experiments/guidelines_log/for_open_source_code/large_ant/learnt/NS_log_1605/"
       
        root_dir="/home/achkan/misc_experiments/guidelines_log/for_open_source_code/large_ant/archive_based//NS_log_34498"

        

        #gen_to_load=range(800,801,1)
        gen_to_load=range(0,1400,1)
        bds_list=set()
      
        append_path=[[0,-24]]#because of the way the sampling of the path is done, the initial point at which the ant is spawned is missing
        append_path=[np.array(x).reshape(1,-1) for x in append_path]
        to_append=np.concatenate(append_path,0)

        for i in gen_to_load:
            if i%10==0:
                print("i==",i)
            with open(root_dir+"/"+f"population_gen_{i}","rb") as fl:
                
                
                agents=pickle.load(fl)
                task_solvers=[x for x in agents if x._solved_task]
                bds=[x._behavior_descr for x in task_solvers]

                if len(task_solvers):
                    print(f"task solved at generation {i}")

                    for x in bds:#bds are of shape 1xN (N is either 32 or 96 up until now)
                        #pdb.set_trace()
                        bds_list.add(tuple(x[0].tolist()))

                    num_pts=int(bds[0].shape[1]//2)
                    bds=[x.reshape(num_pts,2) for x in bds]

                    for bd in bds:
                        bd_full=np.concatenate([bd, to_append], 0)
                        for j in range(len(dummy.env.goals)):
                            goal_j=dummy.env.goals[j]
                            plt.plot(goal_j.coords[0],goal_j.coords[1],color=goal_j.color,marker="o",linestyle="",markersize=25)


                        #plt.plot(bd_full[:,0],bd_full[:,1],"k-")#path
                        #plt.plot(bd_full[-1,0],bd_full[-1,1],"yo")#starting point
                        plt.plot(bd_full[0,0],bd_full[0,1],"mo")#end point
                        #plt.show()
        
        plt.xlim(-45,45)
        plt.ylim(-45,45)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("/tmp/cover_fig.png")
        #plt.show()

        M=np.concatenate([np.array(x).reshape(1,32) for x in bds_list],0)
        cov=np.cov(M.transpose())

        eigs_vals, eigs_vecs= np.linalg.eig(cov)

        print(np.linalg.norm(eigs_vals))

        val=1
        for x in eigs_vals:
            val*=x

        print(val)




    
 

