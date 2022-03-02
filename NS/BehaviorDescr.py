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

from abc import ABC, abstractmethod
import numpy as np
import torch

import pdb

import MiscUtils


class BehaviorDescr:
    """
    Mapping between meta-data and behavior space. Also specifies a metric.
    """

    @staticmethod
    def distance(a, b):
        pass

    @abstractmethod
    def extract_behavior(self, x):
        """
        Extracts behavior descriptor either from meta-data (e.g. in the case of engineered or generic descriptors)
        or from observation (in the case of learned behaviors)
        """
        pass

    @abstractmethod
    def get_bd_dims(self):
        pass


class GenericBD(BehaviorDescr):

    def __init__(self, dims, num):
        """
        dims int  number of dims of behavior descriptor space
        num  int  number of points to sample from trajectory
        """
        self.dims = dims
        self.num = num

    @staticmethod
    def distance(a, b):
        return np.linalg.norm(a - b)

    def extract_behavior(self, trajectory):
        """
        samples self.num trajectory points uniformly, only from the self.dims first dimensions of the trajectory
        trajectory np.array of shape M*dims such that dims>=self.dims
        """
        vec = np.zeros([self.num, self.dims])
        assert trajectory.shape[1] >= vec.shape[1], "not enough dims to extract"
        M = trajectory.shape[0]
        N = vec.shape[0]
        rem = M % N
        inds = list(range(M - 1, rem - 1, -(M // N)))
        vec = trajectory[inds, :self.dims]

        assert len(
            inds) == self.num, "wrong number of samples, this shouldn't happen"

        return vec

    def get_bd_dims(self):

        return self.dims * self.num
