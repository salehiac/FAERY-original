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

from abc import ABC, abstractmethod
import time

import torch
import numpy as np
from scoop import futures


class Agent(ABC):
    _num_instances=0#not threadsafe, create agents only in main thread
    @abstractmethod
    def get_flattened_weights(self):
        pass
    @abstractmethod
    def set_flattened_weights(self, w):
        pass
    
    @abstractmethod
    def get_genotype_len(self):
        pass
    
    def __init__(self):
        self._fitness=None
        self._behavior_descr=None
        self._nov=None
        self._idx=Agent._num_instances+1
        Agent._num_instances+=1

        self._solved_task=False
        self._created_at_gen=-1 #to compute age
        self._parent_idx=-1#hacky way of computing bd distance between parent and child
        self._bd_dist_to_parent_bd=-1
        self._age=-1



class Dummy(torch.nn.Module, Agent):
    def __init__(self, in_d, out_d, out_type="list"):
        torch.nn.Module.__init__(self)
        Agent.__init__(self)
        self.in_d=in_d
        self.out_d=out_d

    def forward(self, x):
        return torch.randn(1,self.out_d)[0,:].tolist()

    def get_flattened_weights(self):
        pass
    def set_flattened_weights(self, w):
        pass
    def get_genotype_len(self):
        pass


_non_lin_dict={"tanh":torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}

def get_num_number_params(model, trainable_only=False):
    if trainable_only:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters()) 
    else:
        model_parameters = model.parameters()

    n_p = sum([np.prod(p.size()) for p in model_parameters])

    return n_p

def get_params_sum(model, trainable_only=False):

    with torch.no_grad():
        if trainable_only:
            model_parameters = filter(lambda p: p.requires_grad, model.parameters()) 
        else:
            model_parameters = model.parameters()

        u=sum([x.sum().item() for x in model_parameters])
        return u

def _identity(x):
    """
    because pickle and thus scoop don't like lambdas...
    """
    return x

class SmallFC_FW(torch.nn.Module, Agent):

    def __init__(self, 
            in_d,
            out_d,
            num_hidden=3,
            hidden_dim=10,
            non_lin="tanh",
            use_bn=False,
            output_normalisation=""):
        torch.nn.Module.__init__(self)
        Agent.__init__(self)

        self.mds=torch.nn.ModuleList([torch.nn.Linear(in_d, hidden_dim)])
        
        for i in range(num_hidden-1):
            self.mds.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mds.append(torch.nn.Linear(hidden_dim, out_d))


        self.non_lin=_non_lin_dict[non_lin] 
        self.bn=torch.nn.BatchNorm1d(hidden_dim) if use_bn else _identity

        self.output_normaliser=_non_lin_dict[output_normalisation] if output_normalisation else _identity


    def forward(self, x, return_numpy=True):
        """
        x list
        """
        out=torch.Tensor(x).unsqueeze(0)
        for md in self.mds[:-1]:
            out=self.bn(self.non_lin(md(out)))
        out=self.mds[-1](out)
        out=self.output_normaliser(out)
        return out.detach().cpu().numpy() if return_numpy else out

    def get_flattened_weights(self):
        """
        returns list 
        """
        flattened=[]
        for m in self.mds:
            flattened+=m.weight.view(-1).tolist()
            flattened+=m.bias.view(-1).tolist()

        #assert len(flattened)==get_num_number_params(self, trainable_only=True)
        return flattened
    
    def set_flattened_weights(self, w_in):
        """
        w_in list
        """

        with torch.no_grad():
            assert len(w_in)==get_num_number_params(self, trainable_only=True), "wrong number of params"
            start=0
            for m in self.mds:
                w=m.weight
                b=m.bias
                num_w=np.prod(list(w.shape))
                num_b=np.prod(list(b.shape))
                m.weight.data=torch.Tensor(w_in[start:start+num_w]).reshape(w.shape)
                m.bias.data=torch.Tensor(w_in[start+num_w:start+num_w+num_b]).reshape(b.shape)
                start=start+num_w+num_b

    def zero_out(self):
        """
        debug function
        """
        with torch.no_grad():
            for m in self.mds:
                m.weight.fill_(0.0)
                m.bias.fill_(0.0)
       
    def check_set_get_flattened_weights(self):
        res=[get_params_sum(self)]
       
        z=self.get_flattened_weights()
        
        self.zero_out()
        res.append(get_params_sum(self))

        self.set_flattened_weights(z)
        res.append(get_params_sum(self))

        test_passed= (res[0]==res[2] and res[1]==0)

        assert test_passed, "this shouldn't happen"
        return test_passed

    def get_genotype_len(self):
        return get_num_number_params(self, trainable_only=True)

                

if __name__=="__main__":

    TEST_AGENTS=False
    TEST_WITH_SCOOP=True

    ag=Dummy(5,2)

    if TEST_AGENTS:
        in_dim=4
        out_dim=3
        
        model=SmallFC_FW(in_d=in_dim,
            out_d=out_dim,
            num_hidden=3,
            hidden_dim=5)
        model.eval()

        z=model.check_set_get_flattened_weights()
        batch_sz=2
        t=torch.rand(batch_sz,in_dim)
        out=model(t)

    elif TEST_WITH_SCOOP:
        num_ags=20
        ags=[]
        for i in range(num_ags):
            ags.append(SmallFC_FW(in_d=np.random.randint(1,100),
                out_d=np.random.randint(1,100),
                num_hidden=np.random.randint(2,100),
                hidden_dim=np.random.randint(1,100)))
            ags[-1].eval()
        t1=time.time()
        res=list(futures.map(get_num_number_params, ags))
        t2=time.time()
        print("res==\n",res)
        print("time=",t2-t1,"secs") #well, it seems faster with scoop -n 1 that with more workers. Probably setting the workers in this case is more costly than the execution


