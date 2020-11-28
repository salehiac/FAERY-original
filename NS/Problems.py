from abc import ABC, abstractmethod
import copy
import time
import numpy as np
import pdb

import gym
import gym_fastsim

from scoop import futures
from scoop import shared as scoop_shared
from termcolor import colored
import BehaviorDescr

class Problem(ABC):
    @abstractmethod
    def __call__(self, index):
        pass
    

class HardMaze(Problem):
    def __init__(self, bd_type="generic", max_episodes=2000, display=False):
        """
        bd_type  str  available options are 
                          - generic   based on spatial trajectory, doesn't take orientation into account
                          - learned   Encoder based, not implemented yet
                          - engineerd here, it's a histogram of states 
        """
        super().__init__()
        self.env = gym.make('FastsimSimpleNavigation-v0')
        self.display= display
    
        if(display):
            self.env.enable_display()
            print(colored("Warning: you have set display to True, makes sure that you have launched scoop with -n 1", "magenta",attrs=["bold"]))

        self.max_episodes=max_episodes

        self.bd_type=bd_type
        if bd_type=="generic":
            self.bd_extractor=BehaviorDescr.GenericBD(dims=2,num=1)#dims=2 for position, no orientation, num is number of samples (here we take the last point in the trajectory)
    
    def close(self):
        self.env.close()

    def __call__(self, index):

        obs=self.env.reset()
        ag=scoop_shared.getConst("population")[index]
        fitness=0
        behavior_info=[] 
        for i in range(self.max_episodes):
            if self.display:
                self.env.render()
                time.sleep(0.01)
            
            action=ag(obs)
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



if __name__=="__main__":
   
    import Agents
    test_scoop=True
    #test_scoop=False

    if test_scoop:
        hm=HardMaze(bd_type="generic",max_episodes=2000,display=False)
        num_agents=100
        random_pop=[Agents.Dummy(in_d=5, out_d=2, out_type="list") for i in range(num_agents)]
        scoop_shared.setConst(population=random_pop)
        
        t1=time.time()
        results=list(futures.map(hm, range(num_agents)))
        t2=time.time()
        print("time==",t2-t1,"secs")#on my machine with 24 cores, I get a factor of about 5x when using all cores instead of just one

        for i in range(len(random_pop)):
            ind=random_pop[i]
            ind._fitness=results[i][0]
            ind._behavior_descr=results[i][1]
            print(i, ind._fitness, ind._behavior_descr)

