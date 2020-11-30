from abc import ABC, abstractmethod
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

class Problem(ABC):
    @abstractmethod
    def __call__(self, agent):
        pass
    

class HardMaze(Problem):
    def __init__(self, bd_type="generic", max_episodes=2000, display=False, assets={}):
        """
        bd_type  str      available options are 
                              - generic   based on spatial trajectory, doesn't take orientation into account
                              - learned   Encoder based, not implemented yet
                              - engineerd here, it's a histogram of states 

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

        self.max_episodes=max_episodes

        self.bd_type=bd_type
        if bd_type=="generic":
            self.bd_extractor=BehaviorDescr.GenericBD(dims=2,num=1)#dims=2 for position, no orientation, num is number of samples (here we take the last point in the trajectory)

        self.maze_im=cv2.imread(assets["env_im"]) if len(assets) else None
        self.num_saved=0

    def close(self):
        self.env.close()

    def __call__(self, ag):
        #print("evaluating agent ", ag._idx)

        if hasattr(ag, "eval"):
            ag.eval()

        obs=self.env.reset()
        fitness=0
        behavior_info=[] 
        for i in range(self.max_episodes):
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

            #print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(o), r, info["dist_obj"], str(info["robot_pos"]), str(eo)))
            if ended:
                break
        
        bd=self.bd_extractor.extract_behavior(np.array(behavior_info).reshape(len(behavior_info), len(behavior_info[0]))) if self.bd_type!="learned" else None
        return fitness, bd

    def visualise_bds(self,pop, quitely=True, save_to=""):
        """
        currently only for 2d generic ones of size 1, so bds should be [bd_0, ...] with bd_i of length 2
        """
        if quitely and not(len(save_to)):
            raise Exception("quitely=True requires save_to to be an existing directory")
        pop=list(pop)
        z=[x._behavior_descr for x in pop]
        most_novel_individual_i=np.argmax([x._nov for x in pop])
        z=np.concatenate(z,0)
        real_w=self.env.map.get_real_w()
        real_h=self.env.map.get_real_w()
        z[:,0]=(z[:,0]/real_w)*self.maze_im.shape[1]
        z[:,1]=(z[:,1]/real_h)*self.maze_im.shape[0]
        
        maze_im=self.maze_im.copy()
        for pt_i in range(z.shape[0]): 
            maze_im=cv2.circle(maze_im, (int(z[pt_i,0]),int(z[pt_i,1])) , 3, color=(0,255,0), thickness=-1)
        
        maze_im=cv2.circle(maze_im, (int(z[most_novel_individual_i,0]),int(z[most_novel_individual_i,1])) , 3, (255,0,0), thickness=-1)
        goal=self.env.map.get_goals()[0]
        maze_im=cv2.circle(maze_im, 
                (int(goal.get_y()*self.maze_im.shape[0]/real_h),int(goal.get_x()*self.maze_im.shape[1]/real_w)),
                3, (0,0,0), thickness=-1)

        if not quitely:
            plt.imshow(maze_im)
            plt.show()
        else:
            if len(save_to):
                cv2.imwrite(save_to+"/hardmaze_2d_bd_"+str(self.num_saved)+".png",maze_im)
                self.num_saved+=1
        return maze_im




if __name__=="__main__":
   
    import Agents
    test_scoop=True
    #test_scoop=False

    if test_scoop:
        hm=HardMaze(bd_type="generic",max_episodes=2000,display=False)
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

