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
import sys
from datetime import datetime
import functools
import pdb
import warnings
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import cv2

sys.path.append("../")
from Data import LinnaeusLoader

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

def plot_with_std_band(x,y,std,color="red",hold_on=False):
    """
    x    np array of size N
    y    np array of size N
    std  np array of size N
    """
    plt.plot(x, y, '-', color='gray')
    plt.fill_between(x, y-std, y+std,
                 color=color, alpha=0.2)

    if not hold_on:
        plt.show()

def dump_pickle(fn, obj):
    with open(fn, "wb") as fl:
        pickle.dump(obj, fl)

class colors:
    red=(255,0,0)
    green=(0,255,0)
    blue=(0,0,255)

_non_lin_dict={"tanh":torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid, "selu": torch.selu, "leaky_relu":torch.nn.functional.leaky_relu}
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

        hidden_dim=3*in_d
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

    def weights_to_constant(self,cst):
        with torch.no_grad():
            for m in self.mds:
                m.weight.fill_(cst)
                m.bias.fill_(cst)
    def weights_to_rand(self,d=5):
        with torch.no_grad():
            for m in self.mds:
                #m.weight.fill_(np.random.randn(m.weight.shape)*d)
                #m.bias.fill_(np.random.randn(m.weight.shape)*d)
                #pdb.set_trace()
                m.weight.data=torch.randn_like(m.weight.data)*d
                m.bias.data=torch.randn_like(m.bias.data)*d

       



class convNxN(torch.nn.Module):

    def __init__(self,
            in_c,#num input channels 
            out_c,#num output channels
            ks=3, #kernel size
            stride=1,
            nonlin="relu",
            padding=1,
            bn=True):

        super().__init__()

        self.cnv=torch.nn.Conv2d(in_c,
                out_c,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                bias=True)


        self.non_lin=_non_lin_dict[nonlin] if len(nonlin) else identity
        self.bn=torch.nn.BatchNorm2d(out_c) if bn else identity

    def forward(self,tns):

        out=self.cnv(tns)
        return self.non_lin(self.bn(out))

class SmallAutoEncoder2d(torch.nn.Module):

    def __init__(self, in_h, in_w, in_c=3, emb_sz=8, with_decoder=True):

        super().__init__()

        logh=np.log2(in_h)
        logw=np.log2(in_w)
        if int(logw)!=logw or int(logh)!=logh:
            #that choice was made to avoid having to do additional upsamplings with interpolocate
            #(because all possible output sizes can't easily be obtaind with ConvTranspose2d) and additional 
            #convolutions to correct that
            raise Exception("spatial dims should be powers of two")

        num_divs_by_two=1
        self.r_h=in_h//(2**num_divs_by_two)
        self.r_w=in_w//(2**num_divs_by_two)

        self.size_lst=[(in_h//(2**x), in_w//(2**x)) for x in range(1,num_divs_by_two+1)]


        self.n_f=16
        #use larger strides instead of maxpooling
        self.mdls_e=torch.nn.ModuleList([
            convNxN(in_c, self.n_f,stride=2),
            torch.nn.Linear(self.n_f*self.r_h*self.r_w, emb_sz)
            ])

        self.mdls_d=None
        if with_decoder:
            self.mdls_d=torch.nn.ModuleList([
                torch.nn.Linear(emb_sz, self.n_f*self.r_h*self.r_w),
                torch.nn.ConvTranspose2d(self.n_f, 3, kernel_size=3, stride=2, padding=1),
                ])

        #self.last=convNxN(3, 3, ks=3, stride=1)
        #self.loss= torch.nn.CrossEntropyLoss()
        self.loss= torch.nn.BCELoss()

    

    def forward(self, x, forward_decoder=True):
        out=x.clone()
        bs=x.shape[0]
        for m in self.mdls_e[:-1]:
            out=m(out)
        #print("before code==",out.shape)
        #code=torch.nn.functional.relu6(self.mdls_e[-1](out.view(bs, -1)))
        code=torch.tanh(self.mdls_e[-1](out.view(bs, -1)))
        #code=self.mdls_e[-1](out.view(bs, -1))
        out_d=None
        if self.mdls_d is not None and forward_decoder:
            out_d=torch.relu(self.mdls_d[0](code))
            out_d=out_d.reshape(bs, self.n_f, self.r_h, self.r_w)
            out_d=out
            for m_i in range(1,len(self.mdls_d)):
                m=self.mdls_d[m_i]
                non_lin=torch.relu if m_i<len(self.mdls_d)-1 else identity
                if isinstance(m,torch.nn.ConvTranspose2d):
                    _, cs, hs, ws = out_d.shape
                    out_d=non_lin(m(out_d,output_size=[bs, cs, hs*2, ws*2]))
                else:
                    out_d=non_lin(m(out_d))
        else:
            return code.detach()

        #out_d=self.last(out_d)
        out_d=torch.sigmoid(out_d)#because inputs are normalized in [-1,1]
        #diff=out_d-x
        #loss=(diff.norm()**2)/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
        target=x.clone().detach()
        #pdb.set_trace()
        loss=self.loss(out_d,target)

        return code.detach(), out_d, loss

def train_encoder(autoencoder, train_shuffle=False, iters=100, dataset="linnaeus"):
    """
    dataset   can either be cifar10 or linnaeus
    """
    import torchvision
    import torchvision.transforms as transforms

    batch_sz=128
   
    if dataset=="cifar10":
        
        degrees=20
        transform_train = transforms.Compose(
                [transforms.ToTensor(),
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomAffine(degrees, translate=(0,0.6), scale=None, shear=2, resample=0, fillcolor=0),
                transforms.ToTensor()])
        
        transform_test= transforms.Compose(
                [transforms.ToTensor()])


        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='../../cifar_data', train=True,
                                            download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz,
                                              shuffle=train_shuffle, num_workers=1)
        
        testset = torchvision.datasets.CIFAR10(root='../../cifar_data', train=False,
                                           download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sz,
                                             shuffle=False, num_workers=1)
        
        LR=1e-3
        optimizer=torch.optim.Adam(autoencoder.parameters(), lr=LR)


    if dataset=="linnaeus":
       
       
        degrees=50
        transform_train = transforms.Compose(
                [transforms.ToTensor(),
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.RandomAffine(degrees, translate=(0,1.0), scale=None, shear=3, resample=0, fillcolor=0),
                    transforms.ToTensor()])
                
        transform_test = transforms.Compose(
                [transforms.ToTensor()])
        
        
        trainset=LinnaeusLoader.Linnaeus("../../linnaeus/",train=True,transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz,
                                              shuffle=train_shuffle, num_workers=1)
        
        testset=LinnaeusLoader.Linnaeus("../../linnaeus/",train=False,transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sz,
                                             shuffle=False, num_workers=1)
        
        LR=1e-3
        #optimizer=torch.optim.SGD(autoencoder.parameters(), lr=LR, momentum=0.9)
        optimizer=torch.optim.Adam(autoencoder.parameters(), lr=LR)

    autoencoder.cuda()

           
    loss_hist=[]
    loss_hist_val=[]
    for epoch in range(iters):  # loop over the dataset multiple times

        if epoch==20:
            LR/=2
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
    
        autoencoder.train()
        epc_loss_h=[]
        tqdm_train = tqdm.trange(len(trainloader), desc='', leave=True)
        train_iter=iter(trainloader)
        for batch_i in tqdm_train:
            data = next(train_iter)
            #pdb.set_trace()
            #if batch_i>2:
            #    break
            if dataset=="cifar10":
                inputs, _ = data #second input is labels, I don't care about them
            else:
                inputs = data #second input is labels, I don't care about them
            inputs=inputs.cuda()

            optimizer.zero_grad()
            _, out_d, loss = autoencoder(inputs)

            if 0:#batch_i%300==0:#epoch%50==0:
                for i in range(1):
                    in_data=inputs[i,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                    out_data=out_d[i,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                    #pdb.set_trace()
                    joint=np.concatenate([in_data,out_data],1)
                    with warnings.catch_warnings():
                        plt.imshow(joint)
                        plt.title("train")
                        plt.show()

            loss.backward()
            optimizer.step()

            epc_loss_h.append(loss.item())
            
            tqdm_train.set_description(f"epoch {epoch}/{iters}, train_loss={sum(epc_loss_h)/len(epc_loss_h)}, LR={LR}")
            tqdm_train.refresh()


        
        loss_hist.append(sum(epc_loss_h)/len(epc_loss_h))
      
        autoencoder.eval()
        epc_val_h=[]
        with torch.no_grad():
            tqdm_val = tqdm.trange(len(testloader), desc='', leave=True)
            test_iter=iter(testloader)
            for batch_i in tqdm_val:
                data=next(test_iter)
                #if batch_i>2:
                #    break
                if dataset=="cifar10":
                    inputs, _ = data #second input is labels, I don't care about them
                else:
                    inputs = data #second input is labels, I don't care about them

                inputs=inputs.cuda()

                _, out_d, loss_v = autoencoder(inputs)
                epc_val_h.append(loss_v.item())
                
                if 0:#batch_i%30==0:#epoch%50==0:
                    for i in range(1):
                        in_data=inputs[i,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                        out_data=out_d[i,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                        joint=np.concatenate([in_data,out_data],1)
                        with warnings.catch_warnings():
                            plt.imshow(joint)
                            plt.title("test")
                            plt.show()


               
                tqdm_val.set_description(f"epoch {epoch}/{iters}, val_loss={sum(epc_val_h)/len(epc_val_h)}, LR={LR}")
                tqdm_val.refresh()
                
            loss_hist_val.append(sum(epc_val_h)/len(epc_val_h))
 
        _=plt.figure()
        plt.plot(loss_hist,"r",label="train")
        plt.plot(loss_hist_val,"b",label="test")
        plt.legend(fontsize=8)
        plt.savefig("/tmp/losses_"+str(epoch)+".png")
        plt.close()

        if dataset=="cifar10":
            #torch.save(autoencoder.state_dict(), "/home/achkan/Desktop/tmp_desktop/autoencoder_saves_cifar10/autoencoder_"+str(epoch))
            torch.save(autoencoder.state_dict(), "/tmp/autoencoder_"+str(epoch))
        else:
            #torch.save(autoencoder.state_dict(), "/home/achkan/Desktop/tmp_desktop/autoencoder_saves_linnaeus/autoencoder_"+str(epoch))
            torch.save(autoencoder.state_dict(), "/tmp/autoencoder_"+str(epoch))
        #torch.save(autoencoder, "/tmp/autoencoder_"+str(epoch))

def load_autoencoder(path, w, h, in_c, emb_sz):
    the_model=SmallAutoEncoder2d(in_h=h, in_w=w, in_c=in_c, emb_sz=emb_sz)
    if torch.cuda.is_available():
        the_model.load_state_dict(torch.load(path))
    else:
        the_model.load_state_dict(torch.load(path,map_location="cpu"))
    return the_model


#class torch_timer:
#    def __init__(self,with_cuda):
#        self.with_cuda=with_cuda
#        self.reset()
#    def reset(self):
#        if self.with_cuda:
#            self.start=torch.cuda.Event(enable_timing=True)
#            self.end=torch.cuda.Event(enable_timing=True)
#    def tic(self):
#        if self.with_cuda:
#            self.reset()
#            self.start.record()
#    def toc(self):
#        if self.with_cuda:
#            self.end.record()
#            torch.cuda.synchronize()
#            return self.start.elapsed_time(self.end)
#
def selRoulette(individuals, k, fit_attr=None):
    """
    taken from deap 
    
    Select *k* individuals from the input *individuals* using *k*
    spins of a roulette. The selection is made by looking only at the first
    objective of each individual. The list returned contains references to
    the input *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    .. warning::
       The roulette selection by definition cannot be used for minimization
       or when the fitness can be smaller or equal to 0.
    """

    s_inds = sorted(individuals, key=lambda x:x._nov, reverse=True)
    sum_fits = sum(getattr(ind, "_nov") for ind in individuals)
    chosen = []
    for i in range(k):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in s_inds:
            sum_ += getattr(ind, "_nov")
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen

if __name__=="__main__":

    TEST_CREAT_DIR=False
    TEST_CREATE_AUENCODER=False
    TRAIN_AUTOENCODER_CIFAR10=False
    #TRAIN_AUTOENCODER_CIFAR10=True

    TRAIN_AUTOENCODER_LINNAEUS=False
    #TRAIN_AUTOENCODER_LINNAEUS=True
    
    #TEST_TRAINED_AUTOENCODER_CIFAR10=False
    TEST_TRAINED_AUTOENCODER_CIFAR10=True

    TEST_TRAINED_AUTOENCODER_LINNAEUS=False
    #TEST_TRAINED_AUTOENCODER_LINNAEUS=True

    
    #TEST_TRAINED_AUTOENCODER_WITH_MAZE_CIFAR10=False
    TEST_TRAINED_AUTOENCODER_WITH_MAZE_CIFAR10=True

    TEST_TRAINED_AUTOENCODER_WITH_MAZE_LINNAEUS=False
    #TEST_TRAINED_AUTOENCODER_WITH_MAZE_LINNAEUS=True

    if TEST_CREAT_DIR:
        _=create_directory_with_pid(dir_basename="/tmp/report_1",remove_if_exists=True,no_pid=True)
        dir_path=create_directory_with_pid(dir_basename="/tmp/report_1",remove_if_exists=True,no_pid=False)

    if TEST_CREATE_AUENCODER:

        N=32
        s2=SmallAutoEncoder2d(in_h=N, in_w=N, in_c=3, emb_sz=64)
        with torch.no_grad():
            s2.eval()
            t=torch.rand(2,3,N,N)
            code, reconstruction, loss=s2(t)
            
    if TRAIN_AUTOENCODER_CIFAR10:
        N=32#cifar10 images are 32x32
        s2=SmallAutoEncoder2d(in_h=N, in_w=N, in_c=3, emb_sz=8)
        train_encoder(s2,train_shuffle=True,iters=100,dataset="cifar10")
    
    if TRAIN_AUTOENCODER_LINNAEUS:
        N=64#linnaeus images are 64x64
        s2=SmallAutoEncoder2d(in_h=N, in_w=N, in_c=3, emb_sz=8)
        train_encoder(s2,train_shuffle=True,iters=100,dataset="linnaeus")

    if TEST_TRAINED_AUTOENCODER_CIFAR10:
        import torchvision
        import torchvision.transforms as transforms


        N=32;
        #path="../models/cifar10/autoencoder_50"
        path="/tmp/autoencoder_40"
        #path="/tmp/autoencoder_50"
        ae=load_autoencoder(path, w=N, h=N, in_c=3, emb_sz=8)
        
        transform = transforms.Compose(
                [transforms.ToTensor()])

        testset = torchvision.datasets.CIFAR10(root='../../cifar_data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=True, num_workers=1)
        test_iter=iter(testloader)

        n_tests=3
        ae.eval()
        ae.cuda()
        with torch.no_grad():
            for i in range(n_tests):
                im, _ = next(test_iter)
                im=im.cuda()
                _, out, loss =ae(im)
                print("loss_val==",loss.item())
                in_data=im[0,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                out=out[0,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                result=np.concatenate([in_data,out],1)
                plt.imshow(result)
                plt.show()
    
    if TEST_TRAINED_AUTOENCODER_LINNAEUS:
        import torchvision
        import torchvision.transforms as transforms

        N=64;
        #path="../models/linnaeus/autoencoder_99"
        path="/tmp/autoencoder_99"
        ae=load_autoencoder(path, w=N, h=N, in_c=3, emb_sz=8)
        
        transform_test = transforms.Compose(
                [transforms.ToTensor()])
        

        testset=LinnaeusLoader.Linnaeus("../../linnaeus/",train=False,transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                shuffle=False, num_workers=1)


        test_iter=iter(testloader)

        n_tests=3
        ae.eval()
        ae.cuda()
        with torch.no_grad():
            for i in range(n_tests):
                im = next(test_iter)
                im=im.cuda()
                _, out, loss =ae(im)
                print("loss_val==",loss.item())
                in_data=im[0,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                out=out[0,:,:,:].transpose(0,1).transpose(1,2).cpu().detach().numpy()
                result=np.concatenate([in_data,out],1)
                plt.imshow(result)
                plt.show()
    
    
    if TEST_TRAINED_AUTOENCODER_WITH_MAZE_CIFAR10:
       
        N=32; model_path="../models/cifar10/autoencoder_50"
        
        with open("/home/achkan/misc_experiments/guideline_results/hard_maze/learned_bds_and_learned_novelty/meta_observation_samples_names_2.pickle","rb") as f:
            im_paths=pickle.load(f)
        im_paths=["/home/achkan/misc_experiments/guideline_results/hard_maze/learned_bds_and_learned_novelty/meta_observation_samples_2/"+x for x in im_paths]
  
        import random
        im_paths=random.sample(im_paths,k=2)

        ae=load_autoencoder(model_path, w=N, h=N, in_c=3, emb_sz=8)
        code_list=[]
        for im_path in im_paths:

       
            ae.eval()
            ae.cuda()
            
            with torch.no_grad():

                #try with hardmaze
                #im=cv2.imread("../maze_b.png").astype("float")
                im=cv2.imread(im_path).astype("float")
                #b,g,r=cv2.split(im)
                #im=cv2.merge([r,g,b])
                im/=255; 
                im_t=torch.Tensor(im).transpose(1,2).transpose(0,1).unsqueeze(0).cuda()
                im_t_s=torch.nn.functional.interpolate(im_t,(N,N))#,mode="bilinear",algin_corners=True)
                im_t_np=im_t_s.transpose(1,2).transpose(3,2).cpu().detach().numpy()[0]
                code,zz,loss=ae(im_t_s)
                code_list.append(code.cpu().detach().numpy())
                zz_np=zz.transpose(1,2).transpose(3,2).cpu().detach().numpy()[0]
                print("loss=",loss)
                #zz_np-=zz_np.min();zz_np/=zz_np.max();
                #zz_np/=zz_np.max();
                
                plt.imshow(im_t_np);plt.show()
                plt.imshow(zz_np);plt.show()

        code_list=np.concatenate(code_list,0)
        print(code_list.max())
        print(code_list.min())

    if TEST_TRAINED_AUTOENCODER_WITH_MAZE_LINNAEUS:
        
        #N=64; model_path="../models/linnaeus/autoencoder_99"
        N=64; model_path="/tmp/autoencoder_99"
        
        with open("/home/achkan/misc_experiments/guideline_results/hard_maze/learned_bds_and_learned_novelty/meta_observation_samples_names_2.pickle","rb") as f:
            im_paths=pickle.load(f)
        im_paths=["/home/achkan/misc_experiments/guideline_results/hard_maze/learned_bds_and_learned_novelty/meta_observation_samples_2/"+x for x in im_paths]
  
        import random
        im_paths=random.sample(im_paths,k=2)

        ae=load_autoencoder(model_path, w=N, h=N, in_c=3, emb_sz=8)
        code_list=[]
        for im_path in im_paths:

       
            ae.eval()
            ae.cuda()
            
            with torch.no_grad():

                #try with hardmaze
                #im=cv2.imread("../maze_b.png").astype("float")
                im=cv2.imread(im_path).astype("float")
                #b,g,r=cv2.split(im)
                #im=cv2.merge([r,g,b])
                im/=255; 
                im_t=torch.Tensor(im).transpose(1,2).transpose(0,1).unsqueeze(0).cuda()
                im_t_s=torch.nn.functional.interpolate(im_t,(N,N))#,mode="bilinear",algin_corners=False)
                im_t_np=im_t_s.transpose(1,2).transpose(3,2).cpu().detach().numpy()[0]
                code,zz,loss=ae(im_t_s)
                code_list.append(code.cpu().detach().numpy())
                zz_np=zz.transpose(1,2).transpose(3,2).cpu().detach().numpy()[0]
                print("loss=",loss)
                #zz_np-=zz_np.min();zz_np/=zz_np.max();
                zz_np/=zz_np.max();
                
                plt.imshow(im_t_np);plt.show()
                plt.imshow(zz_np);plt.show()

        code_list=np.concatenate(code_list,0)
        print(code_list.max())
        print(code_list.min())


