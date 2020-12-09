import gym, gym_fastsim
import time
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

import scipy.stats as stats
import scipy.special
import math
from collections import namedtuple


sys.path.append("../NS/")
sys.path.append("..")
from NS import MiscUtils
#from NS.Agents import Agent
#from NS import Agents


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



def see_evolution_of_learned_novelty_distribution_hardmaze(root_dir, bn_was_used=True, non_lin_type="leaky_relu"):
    """
    root_dir      directory of an NS experiment (NS_log_{pid}), expected to contain frozen_net.model and learnt_{i}.model for i in range(200)
    """
    
    frozen_net_path=root_dir+"/frozen_net.model"
    learned_model_generations=list(range(0,200,50))
    learned_models_paths=[root_dir+f"/learnt_{i}.model" for i in learned_model_generations]
    #print(learned_models_paths)
    
    display= True
    
    env = gym.make('FastsimSimpleNavigation-v0')
    _=env.reset()
    
    width=int(env.map.get_real_w())
    height=int(env.map.get_real_h())
    
    #sys.argv[1]
    frozen=MiscUtils.SmallEncoder1d(2,
            2,
            num_hidden=5,
            non_lin=non_lin_type,
            use_bn=bn_was_used)
    frozen.load_state_dict(torch.load(frozen_net_path))
    frozen.eval()
    
    num_non_frozen=len(learned_models_paths)
    models=[]
    results=[]
    for i in range(num_non_frozen):
        model=MiscUtils.SmallEncoder1d(2,
                2,
                num_hidden=5,
                non_lin=non_lin_type,
                use_bn=bn_was_used)
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
    
    
    #results_np=np.concatenate(results,1)
    #plt.imshow(results_np)
    #plt.show()
    
    
    uniform_distrib=distrib_utils.uniform_like(results[0])
    jensen_shanon_dists=[]
    for i in range(num_non_frozen):
        jensen_shanon_dists.append(distrib_utils.jensen_shannon(results[i],uniform_distrib))
    
    #plt.plot(jensen_shanon_dists,"b")
    #plt.show()
    
    env.close()

    return jensen_shanon_dists

def evolution_of_age_and_parent_child_distances(root_dir):

    ages=[]
    dists=[]
    for gen in range(10):
        fn=root_dir+f"/population_gen_{gen}"
        with open(fn,"rb") as f:
            pop=pickle.load(f)

        ages.append(np.mean([gen - indv._created_at_gen for indv in pop])) 
        dists.append(np.mean([indv._bd_dist_to_parent_bd for indv in pop]))

    return ages, dists


if __name__=="__main__":

    JS_SINGLE_DIRETORY=False
    JS_MULTIPLE_DIRECTORIES=False

    AGE_AND_DISTANCE_TO_PARENT=True
    
    if JS_SINGLE_DIRETORY:
        js=see_evolution_of_learned_novelty_distribution_hardmaze(sys.argv[1])

    if JS_MULTIPLE_DIRECTORIES:
        import os
        
        root="/home/achkan/misc_experiments/guideline_results/hard_maze/learned_novelty_generic_descriptors/uniformity/"
        Experiment=namedtuple("Experiment","path uses_bn non_lin_type")
        
        list_of_experiments=[Experiment(root+"/exp_1/NS_log_4735", False, "tanh"),
                Experiment(root+"/NS_log_48482/", True, "leaky_relu"),
                Experiment(root+"/NS_log_56907/",True, "leaky_relu")]

        js_evolutions=[]
        for x in list_of_experiments:
            js_evol=see_evolution_of_learned_novelty_distribution_hardmaze(x.path, x.uses_bn, x.non_lin_type)
            js_evolutions.append(js_evol)

        js_evolutions=np.array(js_evolutions)
        m_js_evolutions=js_evolutions.mean(0)
        std_js_evolutions=js_evolutions.std(0)
        MiscUtils.plot_with_std_band(range(len(m_js_evolutions)),m_js_evolutions,std_js_evolutions)

    if AGE_AND_DISTANCE_TO_PARENT:

        root="/tmp/"
        list_of_experiments=[root+"/NS_log_14079/"]

        age_evolutions=[]
        bd_dist_to_parent_evolutions=[]
        for ex in list_of_experiments:
            age_ev, dist_to_parent_bd = evolution_of_age_and_parent_child_distances(ex)
            age_evolutions.append(age_ev)
            bd_dist_to_parent_evolutions.append(dist_to_parent_bd)

        age_evolutions=np.array(age_evolutions)
        m_age=age_evolutions.mean(0)
        std_age=age_evolutions.std(0)

        bd_dist_to_parent_evolutions=np.array(bd_dist_to_parent_evolutions)
        m_bd=bd_dist_to_parent_evolutions.mean(0)
        std_bd=bd_dist_to_parent_evolutions.std(0)
        
        MiscUtils.plot_with_std_band(range(len(m_age)),m_age,std_age)
        MiscUtils.plot_with_std_band(range(len(m_bd)),m_bd,std_bd)

