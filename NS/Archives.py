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
import random
import numpy as np
import pickle
#import time
#import sys
#import os
#import copy
import pdb

import expected_distance

class Archive(ABC):
    """
    Interface for the archive type. 
    """
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def manage_size(self):
        pass

    @abstractmethod
    def update(self, pop):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class ListArchive(Archive):

    def __init__(self, 
            max_size=200, 
            growth_rate=6,
            growth_strategy="random",
            removal_strategy="random"):
        self.max_size=max_size
        self.growth_rate=growth_rate
        self.growth_strategy=growth_strategy
        self.removal_strategy=removal_strategy
        self.container=list()

    def reset(self):
        self.container.clear()

    def update(self, pop, thresh=0, boundaries=[], knn_k=-1):
        if self.growth_strategy=="random":
            r=random.sample(range(len(pop)),self.growth_rate)
            candidates=[pop[i] for i in r[:self.growth_rate]]
        elif self.growth_strategy=="most_novel":
            sorted_pop=sorted(pop, key=lambda x: x._nov)[::-1]#descending order
            #print("archive:  ",[u._nov for u in sorted_pop][:self.growth_rate])
            candidates=sorted_pop[:self.growth_rate]
       
        candidates=[c for c in candidates if c._nov>thresh]
        self.container+=candidates

        if len(self)>=self.max_size:
            self.manage_size(boundaries, population=np.concatenate([x._behavior_descr for x in pop],0).transpose(), knn_k=knn_k)

    def manage_size(self,boundaries=[],population=[],knn_k=-1):
        if self.removal_strategy=="random":
            r=random.sample(range(len(self)),k=self.max_size)
            self.container=[self.container[i] for i in r]
        elif self.removal_strategy=="optimal":

            if population.shape[0]!=2:
                raise Exception("this has only be implemented for 2d bds")

            #the order of concatenation is important for the indexations that come next
            reference_set=np.concatenate([np.concatenate([x._behavior_descr for x in self.container],0).transpose(), population], 1)
           
            archive_sz=len(self.container)
            e_ls=[]
            for l in range(archive_sz):#only elements from the archive can be removed
                #e_l, _ =expected_distance.expectation(reference_set, l, k=knn_k, G=50, space_boundaries=boundaries)
                e_l, _ =expected_distance.expectation_parallel(reference_set, l, k=knn_k, G=50, space_boundaries=boundaries)
                e_ls.append(e_l)

            sorted_ids=np.argsort(e_ls)[::-1]#decreasing residual expecation, so we should remove from the end
            self.container=[self.container[i] for i in sorted_ids[:self.max_size]]
            #pdb.set_trace()
        else:
            raise NotImplementedError("manag_size")

    def dump(self, fn):
        with open(fn,"wb") as f:
            pickle.dump(self.container,f)

    def __len__(self):
        return len(self.container)
    
    def __iter__(self):
        return iter(self.container)

    def __str__(self):
        return str(self.container)



if __name__=="__main__":
    pass




