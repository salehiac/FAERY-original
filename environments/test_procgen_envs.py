

import procgen
import gym
import torch

import time

debug_env_seeds=[1053831044, 133241156]
interesting_envs=[128209398]#this one has an enemy and is elongated towards the left


class Conv3x3(torch.nn.Module):

    def __init__(self,
            in_c,
            out_c,
            stride=1,
            nonlin=True,
            tanh=False):

        super().__init__()

        self.use_tanh=tanh
        
        self.cnv=torch.nn.Conv2d(in_c,
                out_c,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True)

        self.bn=torch.nn.BatchNorm2d(out_c) if nonlin else None

    def forward(self,tns):

        out=self.cnv(tns)
        if self.bn is not None:
            if not self.use_tanh:
                out=torch.nn.functional.relu(self.bn(out))
            else:
                out=torch.nn.functional.tanh(out)

        return out
 
class TinyController(torch.nn.Module):

    def __init__(self, in_c):
        super().__init__()

        self.mds=torch.nn.ModuleList([
            Conv3x3(in_c,8,stride=1),
            Conv3x3(8,16,stride=1),
            Conv3x3(16,8,stride=1)])

    def forward(self, tns):
        out=tns
        for md in self.mds:
            out=md(out)
        return out




if __name__=="__main__":

    all_envs=list(gym.envs.registry.all())#for some reason pro
    available_procgen_envs=[str(x) for x in all_envs if "procgen" in str(x)]
    print("available_procgen_envs:\n",available_procgen_envs)
    
    
    #env = gym.make('procgen:procgen-maze-v0',render="human")
    #env = gym.make('procgen:procgen-caveflyer-v0',render="human",start_level=0)
    #env = gym.make('procgen:procgen-caveflyer-v0',render="human", distribution_mode="exploration", use_backgrounds=False)
    env = gym.make('procgen:procgen-caveflyer-v0',render="rgb_array", distribution_mode="exploration", use_backgrounds=False)
    obs = env.reset()
    step = 0
    #while True:
    while 1:
        obs, rew, done, info = env.step(env.action_space.sample())
        print(f"step {step} reward {rew} done {done}")
        step += 1
        if done:
            break


    tc=TinyController(3)
    t=torch.rand(bs, 3, 64, 64)
     
