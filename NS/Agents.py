from abc import ABC, abstractmethod
import torch

class Agent(ABC):
    @abstractmethod
    def get_flattened_weights(self):
        pass
    @abstractmethod
    def set_flattened_weights(self, w):
        pass
    def __init__(self):
        self._fitness=None
        self._behavior_descr=None


class Dummy(torch.nn.Module, Agent):
    def __init__(self, in_d, out_d, out_type="list"):
        torch.nn.Module.__init__(self)
        Agent.__init__(self)
        self.in_d=in_d
        self.out_d=out_d

    def forward(self, x):
        return torch.rand(1,self.out_d)[0,:].tolist()

    def get_flattened_weights(self):
        pass
    def set_flattened_weights(self, w):
        pass



#class SmallFC_FW(torch.nn.Module):

if __name__=="__main__":
    ag=Dummy(5,2)
