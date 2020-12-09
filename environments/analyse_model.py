import gym, gym_fastsim
import time
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
import scipy.special
import math


sys.path.append("..")
from NS import MiscUtils

import distrib_utils

def get_params_sum(model, trainable_only=False):

    with torch.no_grad():
        if trainable_only:
            model_parameters = filter(lambda p: p.requires_grad, model.parameters()) 
        else:
            model_parameters = model.parameters()

        u=sum([x.sum().item() for x in model_parameters])
        return u

def randomize_weights(net):
    u=[x for x in net.mds]
    with torch.no_grad():
        for m in u:
            m.weight.fill_(0.0)
            m.bias.fill_(0.0)


def plot_normal(mu, sigma):
    """
    mu, sigma (==std) floats 
    """
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()


root_dir=sys.argv[1]
frozen_net_path=root_dir+"frozen_net.model"
learned_model_generations=list(range(0,200,1))
learned_models_paths=[root_dir+f"/learnt_{i}.model" for i in learned_model_generations]
print(learned_models_paths)

display= True

env = gym.make('FastsimSimpleNavigation-v0')
_=env.reset()


width=int(env.map.get_real_w())
height=int(env.map.get_real_h())


#sys.argv[1]
frozen=MiscUtils.SmallEncoder1d(2,
        2,
        num_hidden=5,
        non_lin="leaky_relu",
        use_bn=True)
frozen.load_state_dict(torch.load(frozen_net_path))
frozen.eval()

num_non_frozen=len(learned_models_paths)
models=[]
results=[]
for i in range(num_non_frozen):
    model=MiscUtils.SmallEncoder1d(2,
            2,
            num_hidden=5,
            non_lin="leaky_relu",
            use_bn=True)
    model.load_state_dict(torch.load(learned_models_paths[i]))
    model.eval()
    models.append(model)
    results.append(np.zeros([height, width]))


with torch.no_grad():
    for i in range(height):
        if i%10==0:
            print("i==",i)
        batch=torch.cat([torch.ones(width,1)*i,torch.arange(width).float().unsqueeze(1)],1)
        #print(batch)
        z1=frozen(batch)
        for j in range(num_non_frozen):
            z2=models[j](batch)
            diff=(z2-z1)**2
            diff=diff.sum(1)
    
            results[j][i,:]=np.sqrt(diff.cpu().numpy())

for i in range(len(results)):
    results[i]=np.flip(results[i],0)#because hardmaze axis is inverted
    results[i]=scipy.special.softmax(results[i])


results_np=np.concatenate(results,1)
plt.imshow(results_np)
plt.show()


uniform_distrib=distrib_utils.uniform_like(results[0])
jensen_shanon_dists=[]
for i in range(num_non_frozen):
    jensen_shanon_dists.append(distrib_utils.jensen_shannon(results[i],uniform_distrib))

plt.plot(jensen_shanon_dists,"b")
plt.show()

env.close()
