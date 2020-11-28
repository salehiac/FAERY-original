from abc import ABC, abstractmethod
#import time
#import sys
#import os
#import copy



class Archive(ABC):
    """
    Interface for the archvie type. 
    """
    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def add_element(self):
        pass
    
    @abstractmethod
    def remove_element(self, index):
        pass
    
    @abstractmethod
    def manage_size(self):
        pass

    @abstractmethod
    def update(self, pop):
        pass

