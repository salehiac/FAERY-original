import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
import MiscUtils

if __name__=="__main__":


    global_time_networks=[]
    global_time_archive=[]

    #for descr_dim in [2,4,8, 16, 32, 64]:
    #for descr_dim in range(2,64,2):
    #bd_size_range=[2,4,8,16,32,64,128,256]
    bd_size_range=[2,4,8,16,32,64]
    for descr_dim in bd_size_range:
    
        num_gens=100
        
        pop_sz=100 #note that in archive-based methods, nearest neighbours should be found for both population and archive
        emb_dim=descr_dim*4
        pop_bds=torch.rand(pop_sz,descr_dim)
        
        frozen=MiscUtils.SmallEncoder1d(descr_dim,
                emb_dim,
                num_hidden=3,
                non_lin="leaky_relu",
                use_bn=False)
        frozen.eval()

        learnt=MiscUtils.SmallEncoder1d(descr_dim,
                emb_dim,
                num_hidden=5,
                non_lin="leaky_relu",
                use_bn=False)


        #for i in range(10):#to avoid counting gpu warmup time
        #    frozen(pop_bds)

        time_hist=[]
        frozen.eval()
        optimizer = torch.optim.SGD(learnt.parameters(), lr=1e-2)
        batch_sz=128
       
        #torch.cuda.synchronize() #model should be fast on cpu so nevermind gpu timings

        for i in range(num_gens):
            t1=time.time()
          
            #timing for novelty computation before training
            for batch_i in range(0,pop_bds.shape[0],batch_sz):
                batch=torch.Tensor(pop_bds[batch_i:batch_i+batch_sz])

                with torch.no_grad():
                    learnt.eval()
                    e_frozen=frozen(pop_bds)
                    e_pred=learnt(pop_bds)
                    nov=(e_pred-e_frozen).norm(dim=1)

            #this is how training is done, note that we can further reduce runtime by removing the extra frozen forward passes that we've made before when computing novelty
            learnt.train()
            for i in range(5):
                for batch_i in range(0,pop_bds.shape[0],batch_sz):
                    batch=torch.Tensor(pop_bds[batch_i:batch_i+batch_sz])

                    #with torch.no_grad():
                    #    e_frozen=frozen(pop_bds)
                    #    e_pred=learnt(pop_bds)
                    #    nov=(e_pred-e_frozen).norm(dim=1)
                   
                    if 1:
                        optimizer.zero_grad()
                        learnt.train()
                        e_l=learnt(pop_bds)
                        loss=(e_l-e_frozen).norm()**2
                        loss/=pop_sz
                        loss.backward()
                        optimizer.step()
            t2=time.time()
            time_hist.append(t2-t1)
            #print(time_hist)
                     
        mean_t_nets=np.array(time_hist).mean()
        print(descr_dim,mean_t_nets )
        global_time_networks.append(mean_t_nets)

        
        if 1:
            archive_size=10000
            #archive_size=3000
            knn_k=15
            kdt_bds=np.random.rand(archive_size, descr_dim)
               
            times=[]
            for i in range(num_gens):
                #repartitioned)
                t1=time.time()
                #note that the kdtree has to be created everytime as after adding elements, you can't just reuse the same kdtree  (some cells might have become much more dense, and should be
                kdt = KDTree(kdt_bds, leaf_size=20, metric='euclidean')
                dists, ids=kdt.query(pop_bds, knn_k, return_distance=True)
                t2=time.time()
                times.append(t2-t1)

            mean_t_arch=np.array(times).mean()
            global_time_archive.append(mean_t_arch)
            print(descr_dim, mean_t_arch)

    gt_arc_ms=[x*1000 for x in global_time_archive]
    gt_net_ms=[x*1000 for x in global_time_networks]
    #plt.plot(range(0,62,2),gt_arc_ms,"r",label="Archive-based NS",linewidth=5);
    #plt.plot(range(0,62,2),gt_net_ms,"b",label="BR-NS",linewidth=5);
    plt.plot(bd_size_range,gt_arc_ms,"r",label=f"Archive-based NS (size=={archive_size})",linewidth=5);
    plt.plot(bd_size_range,gt_net_ms,"b",label="BR-NS",linewidth=5);
    plt.grid("on");plt.legend(fontsize=28);
    plt.xlabel("behavior descriptor dimensionality",fontsize=28);
    plt.ylabel("time (ms)", fontsize=28);plt.xticks(fontsize=28);
    plt.yticks(fontsize=28);
    plt.xlim(0,60)

    plt.show()
