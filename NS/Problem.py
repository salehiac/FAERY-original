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
    """
    Attention: because of the way libfastsim (which many experiments are based on) works and particularly the
    absence of proper copy constructors in it, Problem instances should NEVER be deep copied (else, their
    env.map is copied but the C++ code ensures that the goals and illuminated switches are cleared).

    So yeah, a bit of gymnastics will be required in meta-learning parallelisations to avoid passing copies to scoop
    """
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
 
