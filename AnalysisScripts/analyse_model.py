import gym, gym_fastsim
import time
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import os

import scipy.stats as stats
import scipy.special
import math
from collections import namedtuple


sys.path.append("../NS/")
sys.path.append("..")
from NS import MiscUtils
#from NS.Agents import Agent
#from NS import Agents

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


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



def see_evolution_of_learned_novelty_distribution_hardmaze(root_dir, 
        bn_was_used=True,
        non_lin_type="leaky_relu", 
        in_dim=2, 
        out_dim=2, 
        behavior_type="generic_2d",
        batch_producer=None):
    """
    root_dir      directory of an NS experiment (NS_log_{pid}), expected to contain frozen_net.model and learnt_{i}.model for i in range(200)
    """
  
    if bn_was_used:
        raise Exception("bn doesn't make sense 1) for frozen, it will modify it's behavior 2) for the other one, not sure but seems like it will hinder convergence")

    frozen_net_path=root_dir+"/frozen_net.model"
    #learned_model_generations=list(range(0,45,10))
    learned_model_generations=list(range(0,400,10))
    #learned_model_generations=list(range(0,20))
    learned_models_paths=[root_dir+f"/learnt_{i}.model" for i in learned_model_generations]
    #print(learned_models_paths)
    
    display= True
    
    env = gym.make('FastsimSimpleNavigation-v0')
    _=env.reset()
    
    width=int(env.map.get_real_w())
    height=int(env.map.get_real_h())
    
    #sys.argv[1]
    frozen=MiscUtils.SmallEncoder1d(in_dim,
            out_dim,
            num_hidden=3,
            non_lin=non_lin_type,
            use_bn=bn_was_used)
    frozen.load_state_dict(torch.load(frozen_net_path))
    frozen.eval()
    
    num_non_frozen=len(learned_models_paths)
    models=[]
    results=[]
    for i in range(num_non_frozen):
        model=MiscUtils.SmallEncoder1d(in_dim,
                out_dim,
                num_hidden=5,
                non_lin=non_lin_type,
                use_bn=bn_was_used)
        model.load_state_dict(torch.load(learned_models_paths[i]))
        model.eval()
        models.append(model)
        results.append(np.zeros([height, width]))
    

    if behavior_type=="generic_2d":
        with torch.no_grad():
            for i in range(height):
                if i%10==0:
                    print("i==",i)
                batch=torch.cat([torch.ones(width,1)*i,torch.arange(width).float().unsqueeze(1)],1)
                #print(batch)
                z1=frozen(batch)
                #print("=========================batch =========================== \n")
                #print(batch)
                #print("=========================z1=========================== \n",z1)
                #input()
                for j in range(num_non_frozen):
                    z2=models[j](batch)
                    diff=(z2-z1)**2
                    diff=diff.sum(1)
                    if torch.isnan(diff).any():
                        print("j==",j)
                        pdb.set_trace()
            
                    results[j][i,:]=np.sqrt(diff.cpu().numpy())

    elif behavior_type=="from_encoder":
        if batch_producer is None:
            raise Exception("you must provide a batch_producer if using from_encoder")
    
    for i in range(len(results)):
        results[i]=np.flip(results[i],0)#because hardmaze axis is inverted
        results[i]=scipy.special.softmax(results[i])
        #results[i]=results[i]/results[i].sum()
    
   
    #pdb.set_trace()
    #results_np=np.concatenate(results,1)
    #plt.imshow(results_np)
    #plt.show()
    #pdb.set_trace()
    
    
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
    for gen in range(0,100,5):
        if gen%100==0:
            print("gen==",gen)
        fn=root_dir+f"/population_gen_{gen}"
        with open(fn,"rb") as f:
            pop=pickle.load(f)

        #pdb.set_trace()
        ages.append(np.mean([gen-indv._created_at_gen for indv in pop])) 
        dists.append(np.mean([indv._bd_dist_to_parent_bd for indv in pop]))

    return ages, dists


if __name__=="__main__":

    JS_SINGLE_DIRETORY=False
    JS_MULTIPLE_DIRECTORIES=False
    AGE_AND_DISTANCE_TO_PARENT=True
    
    if JS_SINGLE_DIRETORY:
        js=see_evolution_of_learned_novelty_distribution_hardmaze(sys.argv[1], bn_was_used=True, non_lin_type="leaky_relu", in_dim=2, out_dim=2)
        plt.plot(js);plt.show()

    if JS_MULTIPLE_DIRECTORIES:
        import os
        
        #root="/home/achkan/misc_experiments/guideline_results/hard_maze/learned_novelty_generic_descriptors/uniformity/"
        #root="/home/achkan/misc_experiments/guidelines_log/"
        #root="/home/achkan/misc_experiments/guidelines_log/learned_novelty/hardmaze_2ddescr/"
        
        Experiment=namedtuple("Experiment","path uses_bn non_lin_type in_dim out_dim")
       
        #root="/tmp/"
        #NS_log_3351
        #list_of_experiments=[Experiment(root+"/NS_log_3351", False, "tanh", 2, 2)]
        #list_of_experiments=[Experiment(root+"/NS_log_67927", True, "leaky_relu", 2, 2)]
        
       
        root="/home/achkan/misc_experiments/guidelines_log/learned_novelty/hardmaze2d/num_optim_iter_5_with_selBest/"
        experiment_names=os.listdir(root)
        list_of_experiments=[]
        for x in experiment_names:
            if "NS_log_" not in x:
                continue
            else:
                list_of_experiments.append(Experiment(root+"/"+x, False, "leaky_relu", 2, 4))

        #list_of_experiments=[Experiment(root+"/NS_log_1058", False, "leaky_relu", 2, 4)]



        #list_of_experiments=[Experiment(root+"/exp_1/NS_log_4735", False, "tanh", 2, 2),
        #        Experiment(root+"/NS_log_48482/", True, "leaky_relu", 2 ,2 ),
        #        Experiment(root+"/NS_log_56907/",True, "leaky_relu", 2, 2)]
        
        #list_of_experiments=[
        #        Experiment(root+"/NS_log_22022/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_24029/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_24980/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_25764/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_26559/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_27382/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_29921/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_32611/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_34345/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_36944/",True, "leaky_relu", 2, 2),
        #        Experiment(root+"/NS_log_39017/",True, "leaky_relu", 2, 2)]

        js_evolutions=[]
        for x in list_of_experiments:
            js_evol=see_evolution_of_learned_novelty_distribution_hardmaze(x.path, x.uses_bn, x.non_lin_type, x.in_dim, x.out_dim)
            js_evolutions.append(js_evol)

        js_evolutions=np.array(js_evolutions)
        m_js_evolutions=js_evolutions.mean(0)
        std_js_evolutions=js_evolutions.std(0)
        MiscUtils.plot_with_std_band(range(len(m_js_evolutions)),m_js_evolutions,std_js_evolutions)

    if AGE_AND_DISTANCE_TO_PARENT:

        #root="/tmp/"
        #list_of_experiments=[root+"/NS_log_63774/"]


        #root="/home/achkan/misc_experiments/guidelines_log/archive_based/hardmaze_8d/"
        #
        #
        #list_of_experiments=[root+"/NS_log_37456/",
        #        root+"NS_log_42210/",
        #        root+"NS_log_43474/",
        #        root+"NS_log_46048/",
        #        root+"NS_log_51056/",
        #        root+"NS_log_55874/",
        #        root+"NS_log_57136/",
        #        root+"NS_log_58441/",
        #        root+"NS_log_59621/",
        #        root+"NS_log_61012/",
        #        root+"NS_log_68822/",
        #        root+"NS_log_77562/"]


        
        #root="/home/achkan/misc_experiments/guidelines_log/learned_novelty/hardmaze_8ddescr/"
        #list_of_experiments=[root+"NS_log_103925/",
        #        root+"NS_log_119372/",
        #        root+"NS_log_63774/",
        #        root+"NS_log_69984/"]
                #root+"NS_log_71510/",
                #root+"NS_log_72894/",
                #root+"NS_log_76509/",
                #root+"NS_log_80616/",
                #root+"NS_log_81854/",
                #root+"NS_log_84993/",
                #root+"NS_log_89880/",
                #root+"NS_log_91040/",
                #root+"NS_log_92328/",
                #root+"NS_log_93631/"]



        #root="/home/achkan/misc_experiments/guidelines_log/hardmaze_2ddescr/"
        #list_of_experiments=[
        #        root+"/NS_log_22022/",
        #        root+"/NS_log_24029/",
        #        root+"/NS_log_24980/",
        #        root+"/NS_log_25764/",
        #        root+"/NS_log_26559/",
        #        root+"/NS_log_27382/",
        #        root+"/NS_log_29921/",
        #        root+"/NS_log_32611/",
        #        root+"/NS_log_34345/",
        #        root+"/NS_log_36944/",
        #        root+"/NS_log_39017/"]

        #root="/home/achkan/misc_experiments/guidelines_log/learned_novelty/hardmaze2d/num_optim_iter_5/"
        root="/home/achkan/misc_experiments/guidelines_log/ant/32d-bd/"
        list_of_experiments=os.listdir(root)
        list_of_experiments=[root+"/"+x for x in list_of_experiments]


        age_evolutions=[]
        bd_dist_to_parent_evolutions=[]
        for ex_i in range(len(list_of_experiments)):
            ex=list_of_experiments[ex_i]
            print(f"===========  experiment {ex_i}/{len(list_of_experiments)} ==========")
            age_ev, dist_to_parent_bd = evolution_of_age_and_parent_child_distances(ex)
            age_evolutions.append(age_ev)
            bd_dist_to_parent_evolutions.append(dist_to_parent_bd)

        age_evolutions=np.array(age_evolutions)
        m_age=age_evolutions.mean(0)
        std_age=age_evolutions.std(0)

        bd_dist_to_parent_evolutions=np.array(bd_dist_to_parent_evolutions)
        m_bd=bd_dist_to_parent_evolutions.max(0)
        std_bd=bd_dist_to_parent_evolutions.std(0)
        
        MiscUtils.plot_with_std_band(range(len(m_age)),m_age,std_age,color="blue",hold_on=True,label="BR-NS")
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.xlabel("generations",fontsize=28)
        plt.ylabel("mean population age",fontsize=28)
        
        if 0:
            archive_age=np.load("/home/achkan/misc_experiments/guidelines_log/archive_based/hardmaze_2d/hardmaze_archived_based_2d_age_evolution.npy")
            MiscUtils.plot_with_std_band(range(len(m_age)),archive_age.mean(0),archive_age.std(0),color="red",hold_on=False,label="Archive-based NS (size=10000)")
        else:
            plt.show()


        MiscUtils.plot_with_std_band(range(len(m_bd)),m_bd,std_bd,color="blue",hold_on=True,label="BR-NS")
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.xlabel("generations",fontsize=28)
        plt.ylabel("mean distance to parent in behavior space",fontsize=28)
        plt.legend(fontsize=28)
        if 0:
            archive_bd_dist_to_par=np.load("/home/achkan/misc_experiments/guidelines_log/archive_based/hardmaze_2d/hardmaze_archived_based_2d_parent_offspring_dist_evolution.npy")
            MiscUtils.plot_with_std_band(range(len(m_age)),archive_bd_dist_to_par.mean(0),archive_bd_dist_to_par.std(0),color="red",hold_on=False,label="Archive-based NS (size=10000)")
        else:
            plt.show()


       

