from abc import ABC, abstractmethod
import random
#import time
#import sys
#import os
#import copy



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

    def update(self, pop):
        if self.growth_strategy=="random":
            r=random.sample(range(len(pop)),range(self.growth_rate))
            self.container+=[pop[i] for i in r[:self.growth_rate]]
        elif self.growth_strategy=="most_novel":
            sorted_pop=sorted(pop, key=lambda x: x._nov)[::-1]#descending order
            self.container+=[pop[i] for i in sorted_pop[:self.growth_rate]]

        if len(self)>=self.max_size:
            self.manage_size()

    def manage_size(self):
        if self.removal_strategy=="random":
            r=random.sample(range(len(self)),k=self.max_size)
            self.container=[self.container[i] for i in r]
        else:
            raise NotImplementedError("manag_size")

    def __len__(self):
        return len(self.container)

    def __str__(self):
        return str(self.container)

class KdTreeBasedArchive(Archive):
    def __init__(self):
        raise NotImplementedError("not implemented")


if __name__=="__main__":

    la=ListArchive(max_size=3)
    for i in range(5):
        la.update("ag",i)
    print(la)




