from abc import ABC, abstractmethod
import torch
import numpy as np

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

class SmallFC_FW(torch.nn.Module, Agent):

    def __init__(self, 
            in_d,
            out_d,
            num_hidden=3,
            hidden_dim=10,
            non_lin="tanh",
            use_bn=False):
        torch.nn.Module.__init__(self)
        Agent.__init__(self)

        self.mds=torch.nn.ModuleList([torch.nn.Linear(in_d, hidden_dim)])
        
        for i in range(num_hidden-1):
            self.mds.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mds.append(torch.nn.Linear(hidden_dim, out_d))


        self.non_lin=_non_lin_dict[non_lin] 
        self.bn=torch.nn.BatchNorm1d(hidden_dim) if use_bn else lambda x: x


    def forward(self, x):
        out=x
        for md in self.mds:
            out=self.bn(self.non_lin(md(out)))
        return out

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

                

if __name__=="__main__":
    ag=Dummy(5,2)

    in_dim=4
    out_dim=3
    model=SmallFC_FW(in_d=in_dim,
            out_d=out_dim,
            num_hidden=3,
            hidden_dim=5)

    z=model.check_set_get_flattened_weights()

    batch_sz=2
    t=torch.rand(batch_sz,in_dim)

    out=model(t)
