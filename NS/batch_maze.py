"""
This file should be removed
"""

import time
import sys
import os
from abc import ABC, abstractmethod
import copy
import functools
import random
import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch
import deap
from deap import tools as deap_tools
from scoop import futures
import yaml
import argparse
from termcolor import colored
import tqdm

import Archives
import NS
import HardMaze
import NoveltyEstimators
import Agents
import MiscUtils

from termcolor import colored
import common_config
if common_config.config_ is not None:
    np.random.seed(common_config.config_.seed)
    random.seed(common_config.config_.seed)
    torch.manual_seed(common_config.config_.seed)
else:
    seed_msg=f"[WARNING] {__file__}: no manual seed. If using meta-world, that could be problematic for behavior repeatability as tasks sampled by metaworld change depending on the seed.\n"
    seed_msg+="If needed, you can use Agent._task_info to retrieve the task and the seeds an agent has succeeded in." 
    print(colored(seed_msg, "green",attrs=["bold"],on_color="on_grey"))


if __name__=="__main__":
    
    #please don't abuse the parser. Algorithmic params should be set in the yaml files
    parser = argparse.ArgumentParser(description='Batch Maze')
    parser.add_argument('--xml_template', type=str,  help="", default="",required=True)
    parser.add_argument('--pbm_dir', type=str,  help="", default="",required=True)
    parser.add_argument('--use_only_first_env', type=bool,  help="", default=False)
    parser.add_argument('--n_pop', type=int,  help="", default=25)
    parser.add_argument('--off_sz', type=int,  help="", default=25)

    args = parser.parse_args()
    
    pbms=os.listdir(args.pbm_dir)
    pbms=[args.pbm_dir+"/"+x for x in pbms]

    if args.use_only_first_env:
        pbms=pbms[:1]

    maze_instances=[]
    for pbm_i in range(len(pbms)):

        pbm=pbms[pbm_i]
        tmp_xml, err, ret_code=MiscUtils.bash_command(["sed","s#pbm_name_here#"+pbm+"#g",args.xml_template])
        tmp_xml=tmp_xml.decode("utf-8")
        tmp_xml=tmp_xml.replace("goal_y","60")
        tmp_xml=tmp_xml.replace("goal_x","60")
        rand_xml_path="/tmp/"+MiscUtils.rand_string()+".xml"
        with open(rand_xml_path,"w") as fl:
            fl.write(tmp_xml)

        assets={"xml_path": rand_xml_path, "env_im":pbm}

        maze_instances.append(HardMaze.HardMaze(bd_type="generic",max_steps=2000, assets=assets))
        
    
    def make_ag():
        return Agents.SmallFC_FW(in_d=maze_instances[0].dim_obs,
                out_d=maze_instances[0].dim_act,
                num_hidden=3,
                hidden_dim=10,
                output_normalisation="")
    mutator=functools.partial(deap_tools.mutPolynomialBounded,eta=10, low=-1.0, up=1.0, indpb=0.1)
    selector=functools.partial(MiscUtils.selBest,k=args.n_pop,automatic_threshold=False)

    for env_i in range(len(pbms)):
        
        nov_estimator= NoveltyEstimators.ArchiveBasedNoveltyEstimator(k=15)
        arch=Archives.ListArchive(max_size=5000,
                    growth_rate=6,
                    growth_strategy="random",
                    removal_strategy="random")

        ns=NS.NoveltySearch(archive=arch,
                nov_estimator=nov_estimator,
                mutator=mutator,
                problem=maze_instances[env_i],
                selector=selector,
                n_pop=args.n_pop,
                n_offspring=args.off_sz,
                agent_factory=make_ag,
                visualise_bds_flag=1,#log to file
                map_type="scoop",#or "std"
                logs_root="/tmp/",
                compute_parent_child_stats=0)

        #do NS
        nov_estimator.log_dir=ns.log_dir_path
        ns.save_archive_to_file=True
        final_pop, solutions, iter_n=ns(iters=400,stop_on_reaching_task=False, save_checkpoints=0)#save_checkpoints is not implemented but other functions already do its job
 





