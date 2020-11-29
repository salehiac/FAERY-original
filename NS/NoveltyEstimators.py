from abc import ABC, abstractmethod
#import time
#import sys
#import os
#import copy

class NoveltyEstimator(ABC):
    """
    Interface for estimating Novelty
    """
    @abstractmethod
    def __call__(self, idx):
        pass
   
        #getattr(_pop_bds[idx].
        #setattr(_pop_bds[idx], novelty_attr)   etc etc
    @abstractmethod
    def update(self, archive, pop):
        pass


class ArchiveBasedNoveltyEstimator(NoveltyEstimator):
    """
    For now parallelising this is just premature optimisation
    """
    def __init__(self, k):
        self.k=k
        self.archive=None
        self.pop=None

    def update(self, archive, pop):
        self.archive=archive
        self.pop=pop

    def __call__(self, idx):
        """
        idx  int   id of individual from the population
        """
        pass

class LearnedNovelty(NoveltyEstimator):

    def __init__(self):
        pass

    def __call__(self,idx):
        raise NotImplementedError("Learned novelty not implemented yet")


