
import time
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import os

from sklearn.neighbors import KDTree
import scipy.stats as stats
import scipy.special
import math
from collections import namedtuple


sys.path.append("../NS/")
sys.path.append("..")
from NS import MiscUtils

def analyse_cycling_behavior_archive_based(root_dir):
    
    ################################################# load all populations and archives

    generations_to_use=list(range(0,110,1))
    populations=[]#those will actually store behavior descriptors
    archives=[]
    for i in generations_to_use:
        with open(root_dir+"/"+f"population_gen_{i}","rb") as fl:
            agents=pickle.load(fl)
            bds=[x._behavior_descr for x in agents]
            populations.append(np.concatenate(bds, 0)) #so each matrix in populations will be of size pop_sz*bd_dim
        with open(root_dir+"/"+f"archive_{i}","rb") as fl:
            agents=pickle.load(fl)
            bds=[x._behavior_descr for x in agents]
            archives.append(np.concatenate(bds, 0)) #so each matrix in populations will be of size pop_sz*bd_dim

    ################################################# compute novelty evolutions
    
    Qmat=np.zeros([len(learned_model_generations), len(learned_model_generations)])

    for i in range(len(generations_to_use)):
        for j in range(i,len(generations_to_use)):
           
            bds_ij=np.concatenate([populations[i], archives[j]],0)
            kdt = KDTree(bds_ij, leaf_size=20, metric='euclidean')

            dists, ids=self.kdt.query(populations[i], self.k, return_distance=True)
            dists=dists[:,1:]
            novs=dists.mean(1)
            Qmat[i,j]=novs

    return Qmat


def analyse_cycling_behavior_learnt_nov(root_dir, in_dim, out_dim):

   
    ################################################# load all networks 
    frozen_net_path=root_dir+"/frozen_net.model"
    frozen=MiscUtils.SmallEncoder1d(in_dim,
            out_dim,
            num_hidden=3,
            non_lin="leaky_relu",
            use_bn=False)
    frozen.load_state_dict(torch.load(frozen_net_path))
    frozen.eval()

    learned_model_generations=list(range(0,110,1))
    #learned_model_generations=list(range(0,900,10))
    learned_models_paths=[root_dir+f"/learnt_{i}.model" for i in learned_model_generations]
    num_non_frozen=len(learned_models_paths)
    models=[]
    for i in range(num_non_frozen):
        model=MiscUtils.SmallEncoder1d(in_dim,
                out_dim,
                num_hidden=5,
                non_lin="leaky_relu",
                use_bn=False)
        model.load_state_dict(torch.load(learned_models_paths[i]))
        model.eval()
        models.append(model)

    assert len(models)==len(learned_model_generations), "this shouldn't happen"

    

    ################################################# load all populations

    populations=[]#those will actually store behavior descriptors
    for i in learned_model_generations:
        with open(root_dir+"/"+f"population_gen_{i}","rb") as fl:
            agents=pickle.load(fl)
            bds=[x._behavior_descr for x in agents]
            populations.append(np.concatenate(bds, 0)) #so each matrix in populations will be of size pop_sz*bd_dim
    
    ################################################# Now let's compute how each generation's novelty evolves through time (ideally, old populations should never be considered novel again
   
    #this will store at each row i, the evolution of mean population novelty for population i through all generations
    Qmat=np.zeros([len(learned_model_generations), len(learned_model_generations)])
    
  
    assert len(models)==len(populations), "something isn't right"
    thresh_u=10
    with torch.no_grad():
        for i in range(len(learned_model_generations)):#over populations
            if i%10==0:
                print("i==",i)
            batch=torch.Tensor(populations[i])#no need to set a batch size, we consider the entier population to be the batch
            pred_f=frozen(batch)
            for j in range(i,len(learned_model_generations)):#over models
            #for j in range(0,len(learned_model_generations)):#over models
                pred_j=models[j](batch)
                diff=(pred_f-pred_j)**2
                diff=diff.sum(1)#this will be of size pop_sz*1
                #diff[diff>thresh_u]=thresh_u
                novelty=diff.mean().item()
                #novelty=diff.median().item()
                Qmat[i,j]=novelty

            s_i=Qmat[i,:].max()
            #Qmat[i,:]/=s_i


    #Qmat=Qmat / Qmat.sum(1)

    return Qmat

    



if __name__=="__main__":
    
   
    if 0:
        root="/home/achkan/misc_experiments/guidelines_log/ant/32d-bd/"
        experiment_names=os.listdir(root)

        experiment_names.pop(experiment_names.index("NS_log_67193"))

        list_of_experiments=[]
        for x in experiment_names:
            if "NS_log_" not in x:
                continue
            else:
                list_of_experiments.append(root+"/"+x)


        list_of_experiments=list_of_experiments[:1]
   
        qmats=[]
        for i in range(len(list_of_experiments)):
            qm=analyse_cycling_behavior_learnt_nov(list_of_experiments[i], in_dim=32, out_dim=64)
            qmats.append(qm)

        plt.imshow(qm);plt.show()

    if 1:
        
        root="/home/achkan/misc_experiments/guidelines_log/ant/32d-bd/"
        experiment_names=os.listdir(root)


        list_of_experiments=[]
        for x in experiment_names:
            if "NS_log_" not in x:
                continue
            else:
                list_of_experiments.append(root+"/"+x)


        list_of_experiments=list_of_experiments[:1]
   
        qmats=[]
        for i in range(len(list_of_experiments)):
            qm=analyse_cycling_behavior_archive_based(list_of_experiments[i])
            qmats.append(qm)

        plt.imshow(qm);plt.show()





