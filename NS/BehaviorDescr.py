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





