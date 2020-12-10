import cv2
import numpy as np
import torch
import os
import sys

sys.path.append("../NS/")
sys.path.append("..")
from NS import BehaviorDescr
class BatchCreator:

    def __init__(self,  image_dir, bs=128):
        """
        image_dir should ONLY contain images (of meta observations)
        """
        self.encoder=BehaviorDescr.FrozenEncoderBased()

        self.ims=os.listdir(image_dir)
        self.ims=[image_dir+x for x in self.ims]

    def create_batch(self):
        pass



if __name__=="__main__":


    bc=BatchCreator("/tmp/meta_observation_samples/")




 





