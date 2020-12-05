

import procgen
import gym
import torch
import matplotlib.pyplot as plt

import time

debug_env_seeds=[1053831044, 133241156]
interesting_envs=[128209398]#this one has an enemy and is elongated towards the left


def Identity(x):
    return x

class Conv3x3(torch.nn.Module):

    def __init__(self,
            in_c,
            out_c,
            k_sz=3,
            stride=1,
            non_lin="relu",
            bn=False):

        super().__init__()

        self.cnv=torch.nn.Conv2d(in_c,
                out_c,
                kernel_size=k_sz,
                stride=stride,
                padding=1,
                bias=True)

        nonlins={"relu":torch.relu, "tanh":torch.tanh}
        self.non_lin=nonlins[non_lin] if len(non_lin) else Identity
        self.bn=torch.nn.BatchNorm2d(out_c) if bn else Identity

    def forward(self,tns):

        out=self.non_lin(self.bn(self.cnv(tns)))

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



def get_patch(im, d):
    """
    im np array of size HxWxC
    d  size of patch to extract
    """
    w,h,c=im.shape
    return im[h//2-d//2:h//2+d//2, w//2-d//2: w//2+d//2, :]

if __name__=="__main__":

    all_envs=list(gym.envs.registry.all())#for some reason pro
    available_procgen_envs=[str(x) for x in all_envs if "procgen" in str(x)]
    print("available_procgen_envs:\n",available_procgen_envs)
    
    
    #env = gym.make('procgen:procgen-caveflyer-v0',render_mode="rgb_array", distribution_mode="exploration", use_backgrounds=False)
    #env = gym.make('procgen:procgen-caveflyer-v0',render_mode="human", distribution_mode="exploration", use_backgrounds=False)
    #env = gym.make('procgen:procgen-caveflyer-v0',render_mode="human", distribution_mode="exploration", use_backgrounds=False, center_agent=True)
    #env = gym.make('procgen:procgen-caveflyer-v0',render_mode="rgb_array", distribution_mode="exploration", use_backgrounds=False, center_agent=True)
    env = gym.make('procgen:procgen-caveflyer-v0',render_mode="rgb_array", distribution_mode="exploration", use_backgrounds=False, center_agent=True)
    #obs = env.reset()
    step = 0
    #while 1:
    for _ in range(50):
        action=env.action_space.sample()
        print(action)
        obs, rew, done, info = env.step(action)
        patch=get_patch(info["rgb"],200)
        #patch=get_patch(obs, 20)
        plt.imshow(patch);plt.show(block=False)
        plt.pause(0.1)
        print(f"step {step} reward {rew} done {done}")
        step += 1
        if done:
            break


