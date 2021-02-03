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

class Problem(ABC):
    @abstractmethod
    def __call__(self, agent):
        pass
    
    @staticmethod
    def get_behavior_space_boundaries(self):
        """
        If the behavior space is a bounded hypercube, returns an np.array of shape N*2 such that 
        the i-th row corresponds to the (lower, upper) bounds in that dimension.
        If the behavior space doesn't have that structure, it sould return None.
        """
        return None
 
