from abc import ABC, abstractmethod
#import time
#import sys
#import os
#import copy

class NoveltyEstimator(ABC):
    """
    Interface for estimating Novelty
    """
    _archive=None
    _pop_bds=None #current population behavior descriptors
    
    def update(self, archive, pop_bd):
        """
        archive           current archive
        pop_bd            population behavior descriptors
        The reason for this function and the static variables is to prevent passing copies to scoop workers
        """
        NoveltyEstimator._arc=archive
        NoveltyEstimator._pop_bds=pop_bds
    
    @abstractmethod
    def __call__(self, idx, novelty_attr, fitness_attr):
        pass
   
        #getattr(_pop_bds[idx].
        #setattr(_pop_bds[idx], novelty_attr)   etc etc

