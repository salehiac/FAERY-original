# Novelty Search Library.
# Copyright (C) 2020 Sorbonne University
# Maintainer: Achkan Salehi (salehi@isir.upmc.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import pdb

import gym
import gym_fastsim

from scoop import futures
from termcolor import colored
import BehaviorDescr
import MiscUtils
from Problem import Problem

sys.path.append("..")
from environments.archimedean_spiral.archimedean_spiral import ArchimedeanSpiral


class ArchimedeanSpiralProblem(Problem):
    def __init__(self, pb_type="", bd_type="", max_steps=20000, display=False, assets={}):
        """
        """
        super().__init__()
        self.env=ArchimedeanSpiral()
        
        self.dim_obs=1
        self.dim_act=0
        self.display= display
    
        self.max_steps=max_steps

        self.dist_thresh=0
        self.num_gens=0

    def close(self):
        self.env.close()

    def get_bd_dims(self):
        return 2

    def __call__(self, ag):
        """
        agent should be a float value indicating angle
        """

        bd=np.array(self.env.compute_value(ag.phi))
        if np.linalg.norm(bd-np.array(self.env.goal)) < 0.5:
            return 1.0, bd.reshape(1,-1), True
        else:
            return 0.0, bd.reshape(1,-1), False

    def visualise_bds(self,archive, population, quitely=True, save_to="/tmp/"):
        """
        """

        self.env.render(hold_on=True)

        bds=[x._behavior_descr for x in population]
        a_bds= [x._behavior_descr for x in archive] if archive is not None else []

        for x in bds:
            plt.plot(x[0,0], x[0,1], color="blue", marker="o")
        for x in a_bds:
            plt.plot(x[0,0], x[0,1], color="magenta", marker="o")
        plt.savefig(save_to+f"/archimedean_spiral_gen_{self.num_gens}.png")
        plt.close()

        self.num_gens+=1


if __name__=="__main__":

    spiral=ArchimedeanSpiralProblem()

    import Agents
    agent=Agents.Agent1d(min(spiral.env.phi_vals), max(spiral.env.phi_vals))
    agent.phi=10
    _, test_bd, _ = spiral(agent)
    agent._behavior_descr=test_bd
    spiral.visualise_bds(archive=None, population=[agent])

   




