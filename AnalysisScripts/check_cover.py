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
from NS import MiscUtils
from NS import Agents

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def check_cover_deceptive_maze(root_dir,
        num_gens_to_check=400,
        G=6,
        h=600,
        w=600):
    """
    G   grid size, applies to both vertical and horizontal
    """

    grid=np.zeros([G,G])
    ratio_h=h//G
    ratio_w=w//G
    cover_hist=[]
    stride=1
    for i in range(0,num_gens_to_check,stride):
        if i%10==0:
            print("i==",i)
        with open(root_dir+f"/population_gen_{i}","rb") as fl:
            pop=pickle.load(fl)
            #pdb.set_trace()

            for x in pop:
                bd_y, bd_x=x._behavior_descr[0].tolist()
                bd_y=h-bd_y
                h_i=int(bd_x//ratio_h)
                v_i=int(bd_y//ratio_w)
                #grid[h_i,v_i]+=1
                grid[h_i,v_i]=1

            bds_i=[x._behavior_descr for x in pop]
            bds_i=np.concatenate(bds_i,0)
            bds_i[:,1]=600-bds_i[:,1]
            #plt.plot(bds_i[:,0],bds_i[:,1],"*b");
            #print("****************************************************************************")
            #print(bds_i)
            ##plt.axis("equal") 
            #plt.xlim(0,600)
            #plt.ylim(0,600)
            ##plt.grid("on")
            #plt.show()
            #pdb.set_trace()

            cover_i=(grid!=0).sum()/(G*G)
            cover_hist.append(cover_i)
    return grid, cover_hist


if __name__=="__main__":


    if 1:#compute learning_based novelty for hard maze with 2d descriptors
        root="/home/achkan/misc_experiments/guidelines_log/learned_novelty/hardmaze2d/num_optim_iter_5_with_selBest/"
        experiments=[]
        if len(experiments)==0:
            experiments=os.listdir(root)
        experiments=[root+x for x in experiments]

        #experiments.pop(experiments.index("/home/achkan/misc_experiments/guidelines_log/learned_novelty/hardmaze2d/num_optim_iter_5/NS_log_67919"))
        #experiments=experiments[:1]

        evolutions=[] 
        for ex in experiments:
            print("ex==",ex)
            im, cover_hist=check_cover_deceptive_maze(ex, num_gens_to_check=400)
            plt.imshow(im)
            plt.show()
            print(cover_hist[-1])
            evolutions.append(cover_hist)
        
        evolutions=np.array(evolutions)#size num_experiments*generations_sampled
   
        e_m=evolutions.mean(0) 
        e_std=evolutions.std(0) 
        MiscUtils.plot_with_std_band(range(e_m.shape[0]), e_m, e_std**2,hold_on=True,color="blue")
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.xlabel("generation", fontsize=28)
        plt.ylabel("coverage", fontsize=28)

    if 1:#checkout archive-based files computed on distant server
        #archive_based_10x10=np.load("ev_mat.npy")
        
        archive_based_6x6=np.load("/home/achkan/misc_experiments/guidelines_log/archive_based/hardmaze_2d/coverage_6x6.npy")

        a_m=archive_based_6x6.mean(0)
        a_s=archive_based_6x6.std(0)
        MiscUtils.plot_with_std_band(range(a_m.shape[0]),a_m,a_s,color="red")








