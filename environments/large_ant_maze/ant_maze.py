"""
"""

import numpy as np
import os

from gym import utils
from gym.spaces import Dict, Box
from gym.utils import EzPickle
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding

#from collections import namedtuple
from collections import deque

from functools import reduce


class GoalArea:
    def __init__(self, 
            coords,
            color,
            ray):
        """
        coords np array
        color  str
        ray    float
        """
        self.coords=coords
        self.color=color
        self.ray=ray
    def dist(self, x):
        return np.linalg.norm(x - self.coords)

    def solved_by(self, x):
        return self.dist(x) < self.ray

class AntObstaclesBigEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_path, max_ts=5000):
        self.ts = 0
        self.goals=[
                GoalArea(np.array([34,-25]),"yellow", 5),
                GoalArea(np.array([-24,33]),"red", 5),
                GoalArea(np.array([15,15]),"blue", 5),
                GoalArea(np.array([4,-24]),"green", 5)]

        self.max_ts = max_ts
        self.xml_path=xml_path
        self.ts=0
        self._obs_hist=deque(maxlen=30)#to check if the ant is stuck


        mujoco_env.MujocoEnv.__init__(self, self.xml_path , frame_skip=5)#not that the max number of steps displayed in the viewer will be frame_skip*self.max_ts, NOT self.max_ts
        utils.EzPickle.__init__(self, xml_path, max_ts)

        #note: don't add members after the call to MujocoEnv.__init__ as it seems to call step

        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        return [seed]

    def step(self, a):
        """
        Note that you shouldn't return goal successes here as step is going to be called in parallel by many processes/threads via a problem's __call__ unless you want to handle the 
        concurrent access issues. It's easier to just evaluate for sucess inside the aforementioned __call__, this way you don't need the hassle of mutexes etc
        """
        self.do_simulation(a, self.frame_skip)
        planar_position=self.data.qpos[:2]
        
        end_episode = False
        if self.ts > self.max_ts:
            end_episode = True
        self.ts+=1

        reward=0#we only want to use pure exploration
        ob = self._get_obs()

        self._obs_hist.append(ob)
        is_stuck=False
        if len(self._obs_hist)==self._obs_hist.maxlen and np.linalg.norm(reduce(lambda x,y: y-x,self._obs_hist,0))<0.5:
            is_stuck=True
            end_episode=True
        
        return ob, reward, end_episode, dict(x_position=planar_position[0],
                                      y_position=planar_position[1],
                                      is_stuck=is_stuck)
                                      

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qpos[:2] = (qpos[:2] - 5) / 70
        return np.concatenate([
            qpos,
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.ts = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[0] = 4.2
        self.viewer.cam.lookat[1] = 0
        self.viewer.opengl_context.set_buffer_size(4024, 4024)

if __name__=="__main__":
    xml_abs_path="/home/achkan/misc_experiments/guidelines_paper/environments/large_ant_maze/xmls/ant_obstaclesbig2.xml"
    ant=AntObstaclesBigEnv(xml_path=xml_abs_path,max_ts=1000)

    obs=ant.reset()
    for i in range(10000):
        ant.render()
        action=ant.action_space.sample()#action is 8d, apparently
        obs, rew, is_done, info=ant.step(action)
        print("ant time steps ==" ,ant.ts)
        if is_done:
            break
        #if i and i%1000==0:
        #    ant.reset()
        #print(obs)
        #print(rew)
        #print("is_done==",is_done)
        #print(info)

    
