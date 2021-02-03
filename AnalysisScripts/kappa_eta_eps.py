import numpy as np
import sys
import torch
import pickle
from termcolor import colored
import scipy.spatial


sys.path.append("../NS/")
sys.path.append("..")
from NS import MiscUtils

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt


def compute_eta(qm_mat):


    res=[]
    for i in range(qm_mat.shape[0]):
        for j in range(i+1,qm_mat.shape[0]):
            res.append(qm[i,j]/qm[i,i])

    return np.mean(res)

def estimate_epsilon_archive_based(problem_bounds):
    """
    estimates epsilon for archived based NS as specified in the paper
    Assumes the behavior space is a hypercube.

    problem_bounds    np.array of shape N*2 such that problem_bounds[i,:] is the (lower,upper) bounds of the hypercube for dimension i of the behavior space.
    """

    eps=np.linalg.norm(problem_bounds[:,1]-problem_bounds[:,0])
    eps=0.1*eps

    return  eps

def estimate_epsilon_br_ns(root_dir, in_dim, out_dim):
    """
    estimates epsilon for BR-NS as stated in the paper
    """
    
    generations_to_use=range(0,2000,10)#we skip some otherwise pdist is intractable

    frozen=MiscUtils.SmallEncoder1d(in_dim,
            out_dim,
            num_hidden=3,
            non_lin="leaky_relu",
            use_bn=False)
    frozen.load_state_dict(torch.load(root_dir+"/frozen_net.model"))
    frozen.eval()
    
    embeddings=[]
    print("estimating epsilon... It will take a bit of time.")
    for i in generations_to_use:
        with open(root_dir+"/"+f"population_gen_{i}","rb") as fl:
            agents=pickle.load(fl)
            bds=[x._behavior_descr for x in agents]
            batch=np.concatenate(bds, 0)

            with torch.no_grad():
                embeddings.append(frozen(batch))



    embeddings_mat=np.concatenate(embeddings, 0)
   
    dists=scipy.spatial.distance.pdist(embeddings_mat)
    eps=0.1*np.max(dists) 
    return eps
            

def compute_kappa(QM,epsilon):
    """
    Computes the kappa measure as defined in BR-NS

    QM np.array of size g*g where g is the number of generations
       QM[i,j] should contain the novelty of population i computed using the novelty estimator of generation j (which can either be an archive or a learnt novelty estimator)
    """

    res=[]
    for i in range(QM.shape[0]):
        zz=QM[i,i:].argmin()
        res.append(sum(QM[i,i+1:]>zz+epsilon))
     
    return np.mean(res)






if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description='Compute Kappa and Eta metrics from the Q-matrix')
    parser.add_argument('--qmat', help='the Q matrix defined in the BR-NS paper, which can be computed from the tools in cycling_analysis.py', required=True)
    parser.add_argument('--problem_type', help='can be either \"archive_based\" or \"br-ns\"', required=True)
    parser.add_argument('--env', help='can be either \"large_ant\" or \"deceptive_maze\"', required=True)
    parser.add_argument('--epsilon', help=' The epsilon from the BR-NS paper. If not specified, computed automatically (in the case of br-ns, you will also need to set root_dir to compute epsilon)', required=False,type=float)
    parser.add_argument('--root_dir', help='If you want epsilon to be computed for BR-NS, you will need to set this parameter to a directory which contains frozen_net.model', required=False)
    parser.add_argument('--skipped_qm', help='As computing the Q matrix is very time-consuming (especially for large archive sizes, sometimes Q(i,j) is only computed at regular intervales where skipped_qm generations are skipped, if so, you should indicate that here.', required=False, default=1, type=int)
    
    args = parser.parse_args()

    if args.epsilon is None:

        if args.problem_type=="archive_based":

            if args.env=="large_ant":
                lam_limits=np.array([[-50,50]])
                lam_limits=np.repeat(lam_limits, 32, axis=0)
                eps=estimate_epsilon_archive_based(lam_limits)
            elif args.env=="deceptive_maze":
                eps=estimate_epsilon_archive_based(np.array([[0,600],[0,600]]))
            else:
                raise Exception("unkown problem type")

        elif args.problem_type=="br-ns":

            if args.root_dir is None:
                raise Exception("You didn't specifiy epsilon for BR-NS so root_dir is required to compute it")

            else:
                if args.env=="large_ant":
                    eps=estimate_epsilon_br_ns(args.root_dir,32, 64) 
                elif args.env=="deceptive_maze":
                    eps=estimate_epsilon_br_ns(args.root_dir,2, 4) 
                else:
                    raise Exception("unsupported problem type")
    else:
        eps=args.epsilon

            
    if ".npz"==args.qmat[-4:]:
        qmz=np.load(args.qmat)
        qm=qmz["arr_0"]
    else:
        qm=np.load(args.qmat)
    
    kappa_value=compute_kappa(qm,eps)
    eta_val=compute_eta(qm) 


    LOW_LIM=100
    UPPER_LIM=qm.shape[0];
  
    RANDOM=False
    if RANDOM:
        N=5
        #color="blue",
        for qm_i in range(N):
            i=np.random.randint(LOW_LIM,UPPER_LIM);
            plt.plot(qm[i,:],label=f"population at generation {i}",linewidth=5);
            plt.yticks(fontsize=28);
    else:
        lst=[100,400,600]

        for i in lst:
            if args.skipped_qm is not None:
                idx=int(i//args.skipped_qm)
                scaled_range=[x*args.skipped_qm for x in range(qm.shape[0])]
            plt.plot(scaled_range, qm[idx,:],label=f"population at generation {i}",linewidth=5);

    
    #plt.title("eta_value="+str(eta_val)+"   kappa_value="+str(kappa_value))
    plt.xticks(fontsize=28);
    plt.yticks(fontsize=28);
    xlab="Archive generation" if args.problem_type=="archive_based" else "Learnt encoder generation"
    ylab="$Q(i,j)$ (Archive-base)" if args.problem_type=="archive_based" else "$Q(i,j)$ (BR-NS)"
    plt.xlabel(xlab,fontsize=28);
    plt.ylabel(ylab,fontsize=28);
    plt.legend(fontsize=28);
    plt.show()


