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
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
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

            #plt.plot(bds_i[:,0],bds_i[:,1],"or");
            plt.scatter(bds_i[:,0],bds_i[:,1],s=80,facecolors="none",edgecolors="r");
            #print("****************************************************************************")
            #print(bds_i)
            #plt.axis("equal") 
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(0,600)
            plt.ylim(0,600)
            plt.grid("on",alpha=3)
            #if i==370:
            #    #plt.show()
            #    plt.axis("off")
            #    plt.savefig("/tmp/explore.png",transparent=True)
            #    #pdb.set_trace()

            cover_i=(grid!=0).sum()/(G*G)
            cover_hist.append(cover_i)
    
    
    ax=plt.gca();
    ax.set_xlim(0, 600);
    ax.set_ylim(0, 600);
    ax.xaxis.set_major_locator(MultipleLocator(int(600//G)));
    ax.yaxis.set_major_locator(MultipleLocator(int(600//G)));
    plt.grid("on");
    plt.show()
    
    return grid, cover_hist


if __name__=="__main__":


    if 1:#compute learning_based novelty for hard maze with 2d descriptors
        #root="/home/achkan/misc_experiments/guidelines_log/learned_novelty/hardmaze2d/num_optim_iter_5_with_selBest/"
        #root="/tmp/"
        #root="/home/achkan/misc_experiments/guidelines_log/archive_management/expectation_based/"
        root="/home/achkan/misc_experiments/guidelines_log/for_open_source_code/learnt_deceptive_maze/"
        experiments=[]
        if len(experiments)==0:
            experiments=os.listdir(root)
        experiments=[root+x for x in experiments if "NS_log_" in x]
        
        #experiments.pop(experiments.index("/home/achkan/misc_experiments/guidelines_log/learned_novelty/hardmaze2d/num_optim_iter_5/NS_log_67919"))
        #experiments.pop(experiments.index("/home/achkan/misc_experiments/guidelines_log/for_open_source_code/learnt_deceptive_maze/NS_log_112362"))

        print(experiments)
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
        
        if 0:#add learned novelty coverage info computed on distant server
            evolutions_1=np.load("/home/achkan/misc_experiments/guidelines_log/learned_novelty/hardmaze2d/coverage_6x6_hardmaze_2d_learned_nov_part_1.npy")
            evolutions=np.concatenate([evolutions, evolutions_1],0)



        #e_m=evolutions.mean(0) 
        e_m=np.median(evolutions,0)
        e_std=evolutions.std(0) 
        MiscUtils.plot_with_std_band(range(e_m.shape[0]), e_m, e_std**2,hold_on=True,color="blue",label="BR-NS")
        plt.xticks(fontsize=28);plt.yticks(fontsize=28);plt.xlabel("generation", fontsize=28);plt.ylabel("coverage", fontsize=28)
        plt.legend(fontsize=28)

   
    if 1:#checkout archive-based files computed on distant server
        
        archive_based_6x6_200=np.load("/home/achkan/misc_experiments/guidelines_log/archive_based/hardmaze_2d/coverage_6x6_hardmaze_2d_archive_based_arch_size_200.npy")

        #a_m_200=archive_based_6x6_200.mean(0)
        a_m_200=np.median(archive_based_6x6_200,0)
        a_s_200=archive_based_6x6_200.std(0)
        MiscUtils.plot_with_std_band(range(a_m_200.shape[0]),a_m_200,a_s_200**2,color="red",hold_on=False,label="Archive-based NS")

        #archive_based_6x6_10000=np.load("/home/achkan/misc_experiments/guidelines_log/archive_based/hardmaze_2d/coverage_6x6_hardmaze_2d_archive_based_arch_size_10000.npy")
        #a_m_10000=archive_based_6x6_10000.mean(0)
        #a_s_10000=archive_based_6x6_10000.std(0)
        #MiscUtils.plot_with_std_band(range(a_m_10000.shape[0]),a_m_10000,a_s_10000,color="orange",hold_on=False)
    


