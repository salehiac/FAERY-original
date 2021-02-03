import time
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb

import scipy.stats as stats
import scipy.special
import math
from collections import namedtuple

sys.path.append("../NS/")
sys.path.append("..")
from NS import MiscUtils as mu
from NS import Agents

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import distrib_utils

import iisignature as iis

from NDGrid import NDGridUniformNoPrealloc

def check_signatures_cover_and_uniformity(sigs,Gs, compute_uniformity=True):
    """
    sigs should be of shape d*N where N is the number of examples and d is the number of dimensions of the signature

    NOTE: uniformity here is computed w.r.t the explored space itself
    """

    min_vals=sigs.min(1)
    max_vals=sigs.max(1)
    grid=NDGridUniformNoPrealloc(Gs=Gs,dims=sigs.shape[0],lower_bounds=min_vals, higher_bounds=max_vals)
    for i in range(sigs.shape[1]):
        grid.visit_cell(sigs[:,i])

    cover=grid.compute_current_coverage()

    js=-1
    if compute_uniformity:

        rr=[freq for freq in grid.visited_cells.values()]
        rr=np.array(rr)
        rr=rr/rr.sum()
        ud=distrib_utils.uniform_like(rr)

        js=distrib_utils.jensen_shannon(rr, ud)


    return cover, js, grid

    





def compute_all_signatures(root_dir,num_pts=16,append_path=[[0, -24]],num_gens_to_check=900):
    """
    num_pts number of samples per trajectory
    append_path  in case we want to append some predefined points to the path for which the signature is being computed. This is the case 
                 for the large ant maze, where I'm sampling N points starting from the end of the trajectory, which means that the starting point
                 of the ant is often ommited. In that case append_path should be [P], where P=[p_0, p_1] is the starting point of the ant
    """
    append_path=[np.array(x).reshape(1,-1) for x in append_path]
    to_append=np.concatenate(append_path,0)
    sig_list=set()
    seen=[]
    for i in range(0,num_gens_to_check,1):
        if i%10==0:
            print("i==",i)
        with open(root_dir+"/"+f"population_gen_{i}","rb") as fl:
            agents=pickle.load(fl)
            task_solvers=[x for x in agents if x._solved_task]
            bds=[x._behavior_descr.reshape(num_pts,2) for x in task_solvers]


            if len(bds):

                for bd in bds:


                    bd_full=np.concatenate([bd, to_append], 0)

                    if bd_full.tolist() not in seen:
                        seen.append(bd_full.tolist())

                    sig=iis.sig(bd_full,3)
                    #pdb.set_trace()
                    sig_list.add(tuple(sig.tolist()))

    sig_list=list(sig_list) 

    return sig_list, seen



if __name__=="__main__":
    
    num_gens_to_check=1000

    if 1:

        root_archive_based="/home/achkan/misc_experiments/guidelines_log/for_open_source_code/large_ant/archive_based/NS_log_34498/"
        sigs_a, seen_a=compute_all_signatures(root_archive_based,num_gens_to_check=num_gens_to_check)

    if 0:

        #sigs_a=np.load("/tmp/sigs_a_4000.npy")
        sigs_a=np.load("/tmp/sigs_a_6000.npy")

    
    if 1:
        #root_learnt="/home/achkan/misc_experiments/guidelines_log/ant/32d-bd/learnt/NS_log_67193"
        #root_learnt="/home/achkan/misc_experiments/guidelines_log/ant/32d-bd/learnt/NS_log_67193"
        #root_learnt="/home/achkan/misc_experiments/guidelines_log/ant/32d-bd/learnt/NS_log_11048"
        root_learnt="/home/achkan/misc_experiments/guidelines_log/for_open_source_code/large_ant/learnt/NS_log_1605/"
        sigs_l, seen_l=compute_all_signatures(root_learnt,num_gens_to_check=num_gens_to_check)

    if 0:
        sigs_l=np.load("/tmp/sigs_l.npy")
    

    sigs_a=np.concatenate([np.array(x).reshape(1,-1) for x in sigs_a],0)
    sigs_l=np.concatenate([np.array(x).reshape(1,-1) for x in sigs_l],0)

    mu.plot_with_std_band(range(sigs_a.shape[1]),sigs_a.mean(0),sigs_a.std(0),hold_on=True,label="Archive of size 6000", only_positive=False,color="red")
    mu.plot_with_std_band(range(sigs_l.shape[1]),sigs_l.mean(0),sigs_l.std(0),hold_on=True,label="BR-NS", only_positive=False,color="blue")
    plt.show()

    plt.plot(sigs_a.std(0),"r")
    plt.plot(sigs_l.std(0),"b")
    plt.show()


    #cover_a, js_a, grid_a=check_signatures_cover_and_uniformity(sigs_a.transpose(),Gs=[3,3,5,20,15,10])
    #cover_l, js_l, grid_l=check_signatures_cover_and_uniformity(sigs_l.transpose(),Gs=[3,3,5,20,15,10])




