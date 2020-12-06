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


import subprocess
import os
from datetime import datetime
import functools
import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

def get_current_time_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def bash_command(cmd:list):
    """
    cmd  list [command, arg1, arg2, ...]
    """
    #print("****************** EXECUTING *************",cmd)
    #input()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    ret_code=proc.returncode

    return out, err, ret_code

def create_directory_with_pid(dir_basename,remove_if_exists=True,no_pid=False):
    while dir_basename[-1]=="/":
        dir_basename=dir_basename[:-1]
    
    dir_path=dir_basename+str(os.getpid()) if not no_pid else dir_basename
    if os.path.exists(dir_path):
        if remove_if_exists:
            bash_command(["rm",dir_path,"-rf"])
        else:
            raise Exception("directory exists but remove_if_exists is False")
    bash_command(["mkdir", dir_path])
    notif_name=dir_path+"/creation_notification.txt"
    bash_command(["touch", notif_name])
    with open(notif_name,"w") as fl:
        fl.write("created on "+get_current_time_date()+"\n")
    return dir_path

class colors:
    red=(255,0,0)
    green=(0,255,0)
    blue=(0,0,255)

_non_lin_dict={"tanh":torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}
def identity(x):
    """
    because pickle and thus scoop don't like lambdas...
    """
    return x

class SmallEncoder1d(torch.nn.Module):
    def __init__(self, 
            in_d,
            out_d,
            num_hidden=3,
            non_lin="relu",
            use_bn=False):
        torch.nn.Module.__init__(self)

        hidden_dim=2*in_d
        self.mds=torch.nn.ModuleList([torch.nn.Linear(in_d, hidden_dim)])

        for i in range(num_hidden-1):
            self.mds.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mds.append(torch.nn.Linear(hidden_dim, out_d))


        self.non_lin=_non_lin_dict[non_lin] 
        self.bn=torch.nn.BatchNorm1d(hidden_dim) if use_bn else identity

    def forward(self, x):
        """
        x list
        """
        out=torch.Tensor(x)
        for md in self.mds[:-1]:
            out=self.bn(self.non_lin(md(out)))

        return self.mds[-1](out)

class convNxN(torch.nn.Module):

    def __init__(self,
            in_c,#num input channels 
            out_c,#num output channels
            ks=3, #kernel size
            stride=1,
            nonlin="relu",
            bn=True):

        super().__init__()

        self.cnv=torch.nn.Conv2d(in_c,
                out_c,
                kernel_size=ks,
                stride=stride,
                padding=1,
                bias=True)


        self.non_lin=_non_lin_dict[nonlin] if len(nonlin) else identity
        self.bn=torch.nn.BatchNorm2d(out_c) if bn else identity

    def forward(self,tns):

        out=self.cnv(tns)
        return self.bn(self.bn(out))

class SmallAutoEncoder2d(torch.nn.Module):

    def __init__(self, in_h, in_w, in_c=3, emb_sz=64, with_decoder=True):

        super().__init__()

        num_divs_by_two=3
        self.r_h=in_h//(2**num_divs_by_two)
        self.r_w=in_w//(2**num_divs_by_two)

        self.size_lst=[(in_h//(2**x), in_w//(2**x)) for x in range(1,num_divs_by_two+1)]

        self.mdls_e=torch.nn.ModuleList([
            convNxN(in_c, 16),
            torch.nn.MaxPool2d(2), 
            convNxN(16, 48),
            convNxN(48, 64),
            torch.nn.MaxPool2d(2),  
            convNxN(64, 32),
            torch.nn.MaxPool2d(2), 
            convNxN(32, 16 ),
            torch.nn.Linear(16*self.r_h*self.r_w, emb_sz)
            ])
     
        self.mdls_d=None
        if with_decoder:
            self.mdls_d=torch.nn.ModuleList([
                torch.nn.Linear(emb_sz, 16*self.r_h*self.r_w),
                convNxN(16, 32),
                torch.nn.ConvTranspose2d(32,32, 3, stride=2, padding=1),
                convNxN(32, 64),
                torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
                convNxN(64, 48),
                convNxN(48, 16),
                torch.nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1),
                convNxN(16, 16),
                ])
            
            self.last=convNxN(16, in_c, nonlin="", bn=False)#to correct after the last upsampling
    

    def forward(self, x, forward_decoder=True):
        out=x.clone()
        bs=x.shape[0]
        for m in self.mdls_e[:-1]:
            out=m(out)
        #print("before code==",out.shape)
        code=self.mdls_e[-1](out.view(bs, -1))
        out_d=None
        if self.mdls_d is not None and forward_decoder:
            out_d=self.mdls_d[0](code)
            out_d=out_d.reshape(bs, 16, self.r_h, self.r_w)
            for m in self.mdls_d[1:]:
                if isinstance(m,torch.nn.ConvTranspose2d):
                    _, cs, hs, ws = out_d.shape
                    out_d=m(out_d,output_size=[bs, cs, hs*2, ws*2])
                else:
                    out_d=m(out_d)

        #print(out_d.shape)
        out_d=torch.nn.functional.interpolate(out_d,(x.shape[2],x.shape[3]))
        out_d=self.last(out_d)
        loss=(out_d-x).norm()/x.shape[0]

        return code, out_d, loss

def train_encoder_on_cifar10(encoder, train_shuffle=False):
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_sz=1
    
    trainset = torchvision.datasets.CIFAR10(root='../../cifar_data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz,
                                          shuffle=train_shuffle, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='../../cifar_data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sz,
                                         shuffle=False, num_workers=2)


    LR=1e-4
    optimizer=torch.optim.SGD(encoder.parameters(), lr=LR, momentum=0.9)
    #optimizer=torch.optim.Adam(encoder.parameters(), lr=LR)

    encoder.cuda()

    iters=300
    tqdm_gen = tqdm.trange(iters, desc='', leave=True)
           
    loss_hist=[]
    loss_hist_val=[]
    for epoch in tqdm_gen:  # loop over the dataset multiple times

        if epoch==100:
            LR/=2
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
    
        encoder.train()
        epc_loss_h=[]
        for batch_i, data in enumerate(trainloader, 0):
            if batch_i>2:
                break
            inputs, _ = data #second input is labels, I don't care about them
            inputs=inputs.cuda()

            #pdb.set_trace()
            optimizer.zero_grad()
            _, out_d, loss = encoder(inputs)

            if epoch%50==0:
                for i in range(1):
                    in_data=inputs[i,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                    out_data=out_d[i,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                    joint=np.concatenate([in_data,out_data],1)
                    plt.imshow(joint)
                    plt.show()

            loss.backward()
            optimizer.step()

            epc_loss_h.append(loss.item())
        
        loss_hist.append(sum(epc_loss_h)/len(epc_loss_h))
      
        encoder.eval()
        epc_val_h=[]
        with torch.no_grad():
            for batch_i, data in enumerate(testloader, 0):
                if batch_i>2:
                    break
                inputs, _ = data #second input is labels, I don't care about them
                inputs=inputs.cuda()

                _, _, loss_v = encoder(inputs)
                epc_val_h.append(loss_v.item())
                
            loss_hist_val.append(sum(epc_val_h)/len(epc_val_h))
 
        tqdm_gen.set_description(f"epoch {epoch}/{iters}, train_loss={loss_hist[-1]}, val_loss={epc_val_h[-1]}, LR={LR}")
        tqdm_gen.refresh()

        #_=plt.figure()
        #plt.plot(loss_hist,"r",label="train")
        #plt.plot(loss_hist_val,"b",label="test")
        #plt.legend(fontsize=8)
        #plt.savefig("/tmp/losses_"+str(epoch)+".png")
        
        


        




if __name__=="__main__":

    if 0:
        _=create_directory_with_pid(dir_basename="/tmp/report_1",remove_if_exists=True,no_pid=True)
        dir_path=create_directory_with_pid(dir_basename="/tmp/report_1",remove_if_exists=True,no_pid=False)

    if 0:

        N=32
        s2=SmallAutoEncoder2d(in_h=N, in_w=N, in_c=3, emb_sz=64)
        with torch.no_grad():
            s2.eval()
            t=torch.rand(2,3,N,N)
            code, reconstruction, loss=s2(t)
            
    if 1:
        N=32#cifar10 images are 32x32
        s2=SmallAutoEncoder2d(in_h=N, in_w=N, in_c=3, emb_sz=32)
        train_encoder_on_cifar10(s2)


