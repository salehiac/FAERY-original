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

class BehaviorDescr:
    @abstractmethod
    def distance(self, other):
        pass
    @abstractmethod
    def extract_behavior(self, x):
        """
        Extracts behavior descriptor either from meta-data (e.g. in the case of engineered or generic descriptors)
        or from observation (in the case of learned behaviors)
        """
        pass

class GenericBD(BehaviorDescr):
    def __init__(self, dims, num):
        self.vec=np.zeros([num, dims])
        self.dims=dims
        self.num=num

    def distance(self, other):
        return np.linalg.norm(self.v-other.v)

    def extract_behavior(self, trajectory):
        """
        samples self.num trajectory points uniformly, only from the self.dims first dimensions of the trajectory
        trajectory np.array of shape M*dims such that dims>=self.vec.shape[1]
        """
        assert trajectory.shape[1]>=self.vec.shape[1], "not enough dims to extract"
        M=trajectory.shape[0]
        N=self.vec.shape[0]
        inds=list(range(M-1,-1,-M//N))
        self.vec=trajectory[inds,:self.dims]

        return self.vec





