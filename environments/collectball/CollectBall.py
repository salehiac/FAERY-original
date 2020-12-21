""""
modified from jeremy fersula's repo
"""

from collections import deque
import time
import os
import numpy as np
from SimpleNav import SimpleNavEnv
import pyfastsim as fs


class DummyAgent:
    def __init__(self,env):
        self.env=env
    def choose_action(self,state):
        #return self.env.action_space.sample()
        return np.random.rand(3).tolist()

class DiscretizeInterval:
    def __init__(self,
            lower, 
            upper,
            num_buckets):
        
        self.ls=np.linspace(lower, upper, num_buckets+1)

    def __call__(self, x):
        if x<self.ls[0] or x>self.ls[-1]:
            raise Exception("value is outside of [lower, upper]")

        c=0
        for i in range(1,len(self.ls)):
            if x<=self.ls[i]:
                break
            else:
                c+=1
        return c



class CollectBall:
    """
    ===> IMPORTANT NOTE: IT SEEMS THAT THE ROBOT'GOAL SPECIFIED IN THE XML IS NEVER USED, INSTEAD IT IS ASSUMED THAT THE GOAL COINCIDES WITH THE ROBOT'S INITIAL POSITION <<===
    
    
    2 Wheeled robot inside a maze, collecting balls and dropping them into a goal.
    The environment is an additional layer to pyFastSim.

    Default observation space is Box(10,) meaning 10 dimensions continuous vector
        1-3 are lasers oriented -45:0/45 degrees
        4-5 are left right bumpers
        6-7 are light sensors with angular ranges of 50 degrees, sensing balls represented as sources of light.
        8-9 are light sensors with angular ranges of 50 degrees, sensing goal also represented as a source of light.
        10 is indicating if a ball is held
    (edit the xml configuration file in ./pyFastSimEnv if you want to change the sensors)

    Action space is Box(3,) meaning 3 dimensions continuous vector, corresponding to the speed of the 2 wheels, plus
    a 'grabbing value', to indicate whether or not the robot should hold or release a ball. (<0 release, >0 hold)

    x,y are by default bounded in [0, 600].

    Environment mutation corresponds to a translation of the balls + translation and rotation of the initial position
    of the robot at the start of one episode.
    """

    def get_weights(self):
        w = list()
        for i in self.init_balls:
            w.append(i[0])
            w.append(i[1])
        w.append(self.init_pos[0])
        w.append(self.init_pos[1])
        w.append(self.init_pos[2])
        return w

    def set_weights(self, weights):
        tups = list()
        for w in range(0, len(weights)-3, 2):
            tups.append((weights[w], weights[w+1]))
        p0 = weights[-3]
        p1 = weights[-2]
        p2 = weights[-1]
        self.init_balls = tups
        self.init_pos = (p0, p1, p2)

        posture = fs.Posture(*self.env.initPos)
        self.env.robot.set_pos(posture)

    def __init__(self, mut_std=5.0, nb_ball=6, ini_pos=(100, 500, 45), setup="SetupMedium.xml"):
        path = os.path.dirname(__file__) + "../env_assets/" + setup
        self.env = SimpleNavEnv(path)
        self.env.reset()

        self.mut_std = mut_std

        self.env.initPos = ini_pos
        posture = fs.Posture(*ini_pos)
        self.env.robot.set_pos(posture)

        for i in range(10):
            if self.check_validity():
                break
            else:
                self.env.initPos = ((self.env.initPos[0] + np.random.uniform(-10, 10)) % 590 + 5,
                                    (self.env.initPos[1] + np.random.uniform(-10, 10)) % 590 + 5,
                                    self.env.initPos[2])
                posture = fs.Posture(*self.env.initPos)
                self.env.robot.set_pos(posture)

        self.init_pos = self.env.get_robot_pos()
        self.ball_held = -1
        self.pos = (self.env.get_robot_pos()[0], self.env.get_robot_pos()[1])

        self.init_balls = [(self.env.get_robot_pos()[0] + 60 * np.cos((2*np.pi) * i/nb_ball),
                            self.env.get_robot_pos()[1] + 60 * np.sin((2*np.pi) * i/nb_ball)) for i in range(nb_ball)]
        self.balls = self.init_balls.copy()

        self.windows_alive = False

        self.proximity_threshold = 10.0  # min distance required to catch or release ball

        ###debug variables
        self.min_orientation=10000
        self.max_orientation=-10000

    def check_validity(self):
        """
        Take a step in a few directions to see if the robot is stuck or not.
        """
        self.env.reset()
        init_state = self.env.get_robot_pos()

        for i in range(4):
            state, reward, done, info = self.env.step((1, 1))

            if abs(info["robot_pos"][0] - init_state[0]) > 0.2 or abs(info["robot_pos"][1] - init_state[1]) > 0.2:
                posture = fs.Posture(*self.env.initPos)
                self.env.robot.set_pos(posture)
                return True
            new_pos = (self.env.initPos[0], self.env.initPos[1], i*90.0)
            posture = fs.Posture(*new_pos)
            self.env.robot.set_pos(posture)
        return False

    def add_balls(self):
        self.env.map.clear_illuminated_switches()
        self.env.map.add_illuminated_switch(fs.IlluminatedSwitch(1, 8, self.init_pos[0], self.init_pos[1], True))
        for x, y in self.balls:
            self.env.map.add_illuminated_switch(fs.IlluminatedSwitch(0, 8, x, y, True))

    def catch(self):
        if self.ball_held == -1:
            for i, (x, y) in zip(range(len(self.balls)), self.balls):
                if np.sqrt((self.pos[0] - x)**2 + (self.pos[1] - y)**2) < self.proximity_threshold:
                    self.ball_held = i
                    self.balls.remove(self.balls[i])
                    self.add_balls()
                    return 0.1
        return 0.0

    def release(self):
        """
        Important note: it seems that the robot'goal specified in the xml is NEVER used, instead it is assumed that the goal coincides with the robot's initial position
        """
        if self.ball_held != -1:
            self.ball_held = -1
            if np.sqrt((self.pos[0] - self.init_pos[0])**2 + (self.pos[1] - self.init_pos[1])**2) \
               < self.proximity_threshold:
                return 1.0
        return 0.0

    def __call__(self, agent, render=False, use_state_path=False, max_steps=12000, exceed_reward=0):
        self.balls = self.init_balls.copy()
        print(self.balls)
        self.add_balls()
        if render and not self.windows_alive:
            self.env.enable_display()
        state = self.env.reset()
        state.append(0.0)
        if len(agent.choose_action(state)) != 3:
            raise Exception("The current agent returned an action of length != 3. Aborting.")
        done = False

        fitness = 0.0

        is_stuck_x = deque()
        is_stuck_y = deque()

        path = list()

        num_cells=5
        num_orientation_bins=4#to keep the descriptor small
        #num_cells*num_cells*num_orientation_bins tensor such that behavior[i,j,k] is the number of times the agents was in cell i,j while having the orientation k (orientation k is
        #discretized in bins)
        #note that orientation here is in radians, i.e. in [-3.14, 3.14], and that width is in [0, self.env.map.get_real_w()], height in [0, self.env.map.get_real_h()], which
        #for the medium env is 600x600
        behavior=np.zeros([num_cells,num_cells,num_orientation_bins])
        discretizer_angle=DiscretizeInterval(-3.14,3.14,num_orientation_bins)
        discretizer_x=DiscretizeInterval(0,self.env.map.get_real_h(),num_cells)
        discretizer_y=DiscretizeInterval(0,self.env.map.get_real_w(),num_cells)

        count = 0
        while not done:
            if render:
                self.env.render()
                time.sleep(0.01)
            action = agent.choose_action(state)
            holding = action[2] > 0

            #introduce some randomness in the exploration
            eps=0.0
            randval=np.random.rand()
            if randval<eps:
                a1=(np.random.rand()-0.5)*2
                a2=(np.random.rand()-0.5)*2
                state, reward, done, info = self.env.step((a1,a2))
            else:
                state, reward, done, info = self.env.step((action[0]*2.0, action[1]*2.0))
            
            state.append(1.0 if self.ball_held != -1 else 0.0)

            self.pos = (self.env.get_robot_pos()[0], self.env.get_robot_pos()[1])

            reward = 0.0  # default reward is distance to goal

            if holding:
                reward += self.catch()

            if not holding:
                reward += self.release()

            fitness += reward

            #angle=self.env.get_robot_pos()[2]
            #self.min_orientation=self.min_orientation if self.min_orientation < angle else angle
            #self.max_orientation=self.max_orientation if self.max_orientation > angle else angle

            ### this is just stupid 
            #if count % 150 == 0 and 900 >= count > 0:#note that this samples the trajectory (6 samples)
            #    path.append(self.pos[0])
            #    path.append(self.pos[1])

            pos_a, pos_b, angle = self.env.get_robot_pos()
            
            #just because there are small errors from libfastsim that return 3.14000001 etc instead of the max 3.14
            angle=3.14 if angle > 3.14 else angle
            angle=-3.14 if angle < -3.14 else angle

            
            angle_id=discretizer_angle(angle)
            #print(behavior.shape)
            #print(angle, angle_id)
            pose_a_id=discretizer_x(pos_a)
            pose_b_id=discretizer_x(pos_b)

            behavior[pose_a_id, pose_b_id, angle_id]+=1

            if count % 50 == 0 and count >= 900:
                if np.array(is_stuck_x).std() + np.array(is_stuck_y).std() < 10:
                    break

            if len(self.balls) == 0 and count >= 900:
                break

            if len(is_stuck_x) == 200:
                is_stuck_x.popleft()
                is_stuck_x.popleft()
                is_stuck_y.popleft()
                is_stuck_y.popleft()
            is_stuck_x.append(self.pos[0])
            is_stuck_y.append(self.pos[1])

            count += 1
            if count >= max_steps:
                fitness += exceed_reward
                break
        return fitness, behavior.reshape(-1) 

    def get_child(self):
        new_init_pos = ((self.init_pos[0] + np.random.normal(0, self.mut_std)) % 560 + 20,
                        (self.init_pos[1] + np.random.normal(0, self.mut_std)) % 560 + 20,
                        (self.init_pos[2] + np.random.normal(0, self.mut_std)) % 360)
        new_env = CollectBall(self.mut_std, ini_pos=new_init_pos)
        new_balls = list()
        for b in self.init_balls:
            # We try to avoid getting too close to the border
            new_balls.append(((b[0] + np.random.normal(0, self.mut_std)) % 560 + 20,
                              (b[1] + np.random.normal(0, self.mut_std)) % 560 + 20))
        new_env.init_balls = new_balls
        new_env.balls=new_env.init_balls.copy()
        return new_env

    def crossover(self, other):
        new_init_pos = self.init_pos if np.random.uniform(0, 1) < 0.5 else other.init_pos
        new_env = CollectBall(self.mut_std, ini_pos=new_init_pos)
        new_balls = list()
        if len(self.init_balls) >= len(other.init_balls):
            for i in range(len(other.init_balls)):
                new_balls.append(self.init_balls[i] if np.random.uniform(0, 1) < 0.5 else other.init_balls[i])
            new_balls += self.init_balls[len(other.init_balls):].copy()
        else:
            for i in range(len(self.init_balls)):
                new_balls.append(self.init_balls[i] if np.random.uniform(0, 1) < 0.5 else other.init_balls[i])
            new_balls += other.init_balls[len(other.init_balls):].copy()
        new_env.init_balls = new_balls
        return new_env

    def __getstate__(self):
        dic = dict()
        dic["Balls"] = self.init_balls
        dic["Std"] = self.mut_std
        dic["Init_pos"] = self.init_pos
        return dic

    def __setstate__(self, state):
        self.__init__(state["Std"], ini_pos=state["Init_pos"])
        self.init_balls = state["Balls"]

    def __del__(self):
        self.env.close()

if __name__=="__main__":

    cb=CollectBall(mut_std=35.0, ini_pos=(80, 480, 45), nb_ball=8, setup="SetupMedium.xml")
    #for i in range(15):
    #    cb=cb.get_child()
    cb.init_balls=[
            (540,480),
            (60,380),
            (120,210),
            (520,90),
            (100,100),
            (200,200),
            (340,230),
            (400,500)
            ]

    dummy_ag=DummyAgent(cb.env)
    cb.env.enable_display()
    _, beh=cb(dummy_ag,render=True,max_steps=500)


 
