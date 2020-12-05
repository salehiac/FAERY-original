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
from sklearn.neighbors import KDTree
import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb
#import time
#import sys
#import os
#import copy

import MiscUtils

class NoveltyEstimator(ABC):
    """
    Interface for estimating Novelty
    """
    @abstractmethod
    def __call__(self):
        """
        estimate novelty of entire current population w.r.t istelf+archive
        """
        pass
   
        #getattr(_pop_bds[idx].
        #setattr(_pop_bds[idx], novelty_attr)   etc etc
    @abstractmethod
    def update(self, pop, archive=None):
        pass


class ArchiveBasedNoveltyEstimator(NoveltyEstimator):
    """
    For now parallelising this is just premature optimisation
    """
    def __init__(self, k):
        self.k=k
        self.archive=None
        self.pop=None

    def update(self, pop, archive):
        self.archive=archive
        self.pop=pop
 
        self.pop_bds=[x._behavior_descr for x in self.pop]
        self.pop_bds=np.concatenate(self.pop_bds, 0)
        self.archive_bds=[x._behavior_descr for x in self.archive] 
        
        if len(self.archive_bds):
            self.archive_bds=np.concatenate(self.archive_bds, 0) 
       
        self.kdt_bds=np.concatenate([self.archive_bds,self.pop_bds],0) if len(self.archive_bds) else self.pop_bds
        self.kdt = KDTree(self.kdt_bds, leaf_size=20, metric='euclidean')

    def __call__(self, dist_thresh):
        """
        estimate novelty of entire current population w.r.t istelf+archive

        dist_thresh   float   if an individual is closer to its nearest neightbour in archive+pop than dist_thresh, then 
                              it wont be added to the archive (without that condition you might add the same point numerous 
                              times to the archive, as while its distance to its nearset neighbour is ~0, it couls be far from all of
                              its second, third, .... neighbours.

        returns novelties as unsorted list
        """
        dists, ids=self.kdt.query(self.pop_bds, self.k, return_distance=True)
        #the first column is the point itself because the population itself is included in the kdtree
        dists=dists[:,1:]
        ids=ids[:,1:]

        #mask_1=dists[:,0]<dist_thresh
        #mask_2=ids[:,0]<len(self.archive)#this is because I want to consider an individual that is novel relative to the archive but not relative to the population as novel
        #mask=np.logical_and(mask_1,mask_2)
        #dists[mask,0]=np.ones([mask.astype(int).sum()])*float("inf")*-1
        novs=dists.mean(1)
        #pdb.set_trace()
        return novs.tolist()


class LearnedNovelty(NoveltyEstimator):

    def __init__(self, in_dim, emb_dim, batch_sz=128):

        self.frozen=MiscUtils.SmallEncoder(in_dim,
            emb_dim,
            num_hidden=5,
            non_lin="tanh",
            use_bn=False)
        self.frozen.eval()
        
        self.learnt=MiscUtils.SmallEncoder(in_dim,
            emb_dim,
            num_hidden=5,
            non_lin="tanh",
            use_bn=False)
       
        self.optimizer = torch.optim.SGD(self.learnt.parameters(), lr=1e-3)
        self.archive=None
        self.pop=None
        self.batch_sz=batch_sz
        
    def update(self, pop, archive=None):
        
        self.pop=pop
 
        self.pop_bds=[x._behavior_descr for x in self.pop]
        self.pop_bds=np.concatenate(self.pop_bds, 0)
    
    def __call__(self, dist_tresh):

        #self.pop_bds is of size NxD with D the dimensions of the behavior space
      
        pop_novs=[]
        for i in range(0,self.pop_bds.shape[0],self.batch_sz):
            batch=torch.Tensor(self.pop_bds[i:i+self.batch_sz])
            with torch.no_grad():
                e_frozen=self.frozen(batch)
                self.learnt.eval()
                e_pred=self.learnt(batch)
                diff=(e_pred-e_frozen).norm(dim=1)
                pop_novs+=diff.cpu().detach().tolist()
            
            self.learnt.train()
            self.optimizer.zero_grad()
            e_l=self.learnt(batch)
            loss=(e_l-e_frozen).norm()**2
            loss.backward()
            self.optimizer.step()

        assert len(pop_novs)==self.pop_bds.shape[0], "that shouldn't happen"
        return pop_novs

if __name__=="__main__":

    if 0:

        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        kdt = KDTree(X, leaf_size=20, metric='euclidean')
        queries=np.array([[2,2],[5,5],[0.5,1.0]])
        u=kdt.query(queries, k=1, return_distance=False)
        u=u.reshape(queries.shape[0]).tolist()

        plt.plot(X[:,0], X[:,1],"r*")
        plt.plot(queries[:,0],queries[:,1],"b*")
        plt.plot(X[u,0],X[u,1],"mx")
        plt.show()

    if 1:

        ln=LearnedNovelty(2,2)
        ln.pop_bds=np.random.rand(218,2)
        ln()


