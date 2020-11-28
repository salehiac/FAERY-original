from abc import ABC, abstractmethod
import copy
import time

import gym
import gym_fastsim

import BehaviorDesrc 

class Problem(ABC):
    _population=[]#in order to allow scoop to see those without copying N times
    @abstractmethod
    def __call__(self, index):
        pass
    
    @staticmethod
    def set_static_pop(pop):
        Problem._population=copy.deepcopy(pop)

class HardMaze(Problem):
    def __init__(self, str:bd_type="generic", display=False):
        """
        bd_type  str  available options are 
                          - generic   based on spatial trajectory, doesn't take orientation into account
                          - learned   Encoder based, not implemented yet
                          - engineerd here, it's a histogram of states 
        """
        super().__init__(self)
        self.env = gym.make('FastsimSimpleNavigation-v0')
        self.display= display
    
        if(display):
            self.env.enable_display()

        self.max_episodes=200

        self.bd_type=bd_type
        if bd_type=="generic":
            self.bd_extractor=BehaviorDesrc.GenericBD(1,2)
    
    def close(self):
        self.env.close()

    def __call__(self, index):

        obs=self.env.reset()
        ag=HardMaze._population[index]
        fitness=0
        behavior_info=[] 
        for i in range(self.max_episodes):
            if self.display:
                env.render()
        	time.sleep(0.01)
            
            action=ag(obs)
            obs, reward, ended, info=env.step(action)
            fitness+=reward
            if bd_type!="learned":
                behavior_info.append(info["robot_pos"])
            else:
                behavior_info.append(obs)

            #print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(o), r, info["dist_obj"], str(info["robot_pos"]), str(eo)))
            if ended:
                break

        ag._fitness=fitness
        ag._behavior_descr=self.bd_extractor.extract_behavior(behavior_info) if self.bd_type!="learned" else None

     
