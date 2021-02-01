import numpy as np
import matplotlib.pyplot as plt



def compute_eta(qm_mat):


    res=[]
    for i in range(qm_mat.shape[0]):
        for j in range(i+1,qm_mat.shape[0]):
            res.append(qm[i,j]/qm[i,i])

    return np.mean(res)




if __name__=="__main__":

    #qmz=np.load("qmat_learned_4000_iters_with_divergence_training.npz")
    qmz=np.load("/home/achkan/misc_experiments/guidelines_log/for_open_source_code/qm_large_ant_maze_br_ns_run_1.npz")
    qm=qmz["arr_0"]

    #qm=np.load("/home/achkan/misc_experiments/guidelines_log/cycling_behavior/archive_based/qm_archive_4000.npy")
    
   
    LOW_LIM=200
    UPPER_LIM=qm.shape[0];
  
    RANDOM=True
    if RANDOM:
        N=5
        #color="blue",
        for qm_i in range(N):
            i=np.random.randint(LOW_LIM,UPPER_LIM);
            plt.plot(qm[i,:],label=f"population at generation {i}",linewidth=5);
            plt.title(str(i));plt.xticks(fontsize=28);
            plt.yticks(fontsize=28);
    else:
        lst=[300,500,700,900,1100,1300,1500]
        for i in lst:
            plt.plot(qm[i,:],label=f"population at generation {i}",linewidth=5);
            plt.title(str(i));
            plt.xticks(fontsize=28);
            plt.yticks(fontsize=28);


    
    eta_val=compute_eta(qm) 
    plt.title("eta_value="+str(eta_val)+"   kappa_value="+str(4.10510204081632))
    plt.xticks(fontsize=28);
    plt.yticks(fontsize=28);
    plt.xlabel("Archive generation",fontsize=28);
    plt.ylabel("K-nn based Novelty",fontsize=28);
    plt.legend(fontsize=18);
    plt.show()

