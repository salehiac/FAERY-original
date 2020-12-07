import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class Linnaeus(Dataset):

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir  should have two subdirs test and train, that contain NOTHING but jpg images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.train=train
      
        if not self.train:
            self.fls=sorted(os.listdir(root_dir+"/test/"),key=lambda x:int(x.split("_")[0]))
            self.fls=[root_dir+"./test/"+x for x in self.fls]
        else:
            self.fls=sorted(os.listdir(root_dir+"./train"),key=lambda x:int(x.split("_")[0]))
            self.fls=[root_dir+"./train/"+x for x in self.fls]
        
        
        self.transform = transform

    def __len__(self):
        return len(self.fls)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()


        im=cv2.imread(self.fls[idx])
        b, g, r=cv2.split(im)
        im = cv2.merge([r,g,b])
        if self.transform:
            im = self.transform(im)

        return im


if __name__=="__main__":

    ds=Linnaeus("./",train=False)
    print("len==",len(ds))

    #plt.imshow(ds[0]);plt.show()

    batch_sz=4
    trainloader = torch.utils.data.DataLoader(ds, batch_size=batch_sz,shuffle=False, num_workers=1)
    train_iter=iter(trainloader)

    for i in range(5):

        for batch_i in range(batch_sz):
            data=next(train_iter)
            data_np=data.cpu().detach().numpy()[batch_i]
            plt.imshow(data_np);
            plt.title(str(batch_i))
            plt.show()






