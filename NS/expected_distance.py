"""
Should this file be removed?
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scoop import futures

def _compute_terms_for_single_b(args):
    """
    see the expectation function to understand this more easily
    """
    cell_sz, i,j,pop, l,k = args
    b_i=cell_sz*i
    b_j=cell_sz*j

    b=np.array([[b_i],[b_j]])#2x1

    d_l=np.linalg.norm(b-pop[:,l].reshape(2,1))

    dists_b=[]
    for pop_i in range(pop.shape[1]):
        dists_b.append(np.linalg.norm(b-pop[:,pop_i].reshape(2,1)))

    knn_ids=np.argsort(dists_b)[:k]

    if l not in knn_ids:#the expectation is only over the b for which a_l is a k-nn
        return 0,0,+1 #even if its value is 0 it sill sould count

    d_u=sorted(dists_b)[k+1]

    return d_u, d_l, +1



def expectation_parallel(pop, l, k, G, space_boundaries=[]):
    """
    parallel version

    computes, assuming a 2d bounded space, an approximation to 
    E_b[d(a_u, b) - d(a_l, b] 
    where the expectation is over all b for which a_l is one of their k-nns and where
    a_u is the (k+1)-th neighbour of b

    for simplicity here we assume that the bounded space has equal length in both dimensions.
    space_boundaries is thus a  list of size 2: [low_val, high_val]

    G is the number of the cells used in the approximation
    pop is of shape 2xN
    l is in [0,N-1]
    k is the k in knn
    """

    term_0=0#left term
    term_1=0

    space=np.zeros([G,G])
    cell_sz=(space_boundaries[1]-space_boundaries[0])/G

     
    uu=futures.map(_compute_terms_for_single_b,[[cell_sz, i,j,pop,l,k] for i in range(G) for j in range(G)])
    uu=list(uu)
    
    d_us=[x[0] for x in uu]
    d_ls=[x[1] for x in uu]
    num_b_lst=[x[2] for x in uu]

    #pdb.set_trace()

    term_0=sum(d_us)/sum(num_b_lst)
    term_1=sum(d_ls)/sum(num_b_lst)

    #compute the novelty of a_l:
    dists=[]
    for i in range(pop.shape[1]):
        if i==l:
            continue
        dists.append(np.linalg.norm(pop[:,i]-pop[:,l]))
    nearest_ds=np.sort(dists)[:k]
    #nearest_ds=np.sort(dists)
    novelty=nearest_ds.mean()

    #print(f"l={l},        term_0={term_0},        term_1={term_1},                term={term_0-term_1},    novelty_l={novelty}")

    res=term_0-term_1

    return res, novelty




def expectation(pop, l, k, G, space_boundaries=[]):#note that pop is the reference set pop U archive, l is an index in pop
    """
    computes, assuming a 2d bounded space, an approximation to 
    E_b[d(a_u, b) - d(a_l, b] 
    where the expectation is over all b for which a_l is one of their k-nns and where
    a_u is the (k+1)-th neighbour of b

    for simplicity here we assume that the bounded space has equal length in both dimensions.
    space_boundaries is thus a  list of size 2: [low_val, high_val]

    G is the number of the cells used in the approximation
    pop is of shape 2xN
    l is in [0,N-1]
    k is the k in knn
    """

    term_0=0#left term
    term_1=0

    space=np.zeros([G,G])
    cell_sz=(space_boundaries[1]-space_boundaries[0])/G

    num_bs=0#this wont be the same for all a_l
    for i in range(G):
        for j in range(G):
            b_i=cell_sz*i
            b_j=cell_sz*j

            b=np.array([[b_i],[b_j]])#2x1

            d_l=np.linalg.norm(b-pop[:,l].reshape(2,1))

            dists_b=[]
            for pop_i in range(pop.shape[1]):
                dists_b.append(np.linalg.norm(b-pop[:,pop_i].reshape(2,1)))

            knn_ids=np.argsort(dists_b)[:k]
            if l not in knn_ids:#the expectation is only over the b for which a_l is a k-nn
                num_bs+=1 #this is an expectation, so even if the value is zero we should count it as a value in the expectation
                continue

            d_u=sorted(dists_b)[k+1]

            term_0+=d_u
            term_1+=d_l
            num_bs+=1

    term_0/=(num_bs+1e-9)
    term_1/=(num_bs+1e-9)

    #compute the novelty of a_l:
    dists=[]
    for i in range(pop.shape[1]):
        if i==l:
            continue
        dists.append(np.linalg.norm(pop[:,i]-pop[:,l]))
    nearest_ds=np.sort(dists)[:k]
    #nearest_ds=np.sort(dists)
    novelty=nearest_ds.mean()

    #print(f"l={l},        term_0={term_0},        term_1={term_1},                term={term_0-term_1},    novelty_l={novelty}")

    res=term_0-term_1

    return res, novelty

if __name__=="__main__":


    check_expectation_for_same_l=False
    check_different_ls=True
    num_experiments=5

    if check_expectation_for_same_l:
        results=[]
        for exp_i in range(num_experiments):
            if exp_i%10==0:
                print("experiment ",exp_i)
            high_b=60
            low_b=0

            num_pop=15
            x=np.random.rand(2,num_pop)*high_b
            res=expectation(x, l=5, k=3, G=100, space_boundaries=[0, high_b])
            results.append(res)
            #print(res)
            #plt.plot(x[1,:],x[0,:],"ro");plt.show()
        plt.plot(results,"ro");plt.show()

    if check_different_ls:

        has_zero=[]
        for exp_i in range(num_experiments):
            if exp_i%10==0:
                print("experiment ",exp_i)

            results=[]
            novs=[]
            
            high_b=60
            low_b=0

            num_pop=20
            #x=np.random.rand(2,num_pop)*high_b
            x=np.random.rand(2,num_pop)*15+10
            #plt.plot(x[1,:],x[0,:],"ro");plt.show()

            k_val=8
            for p_i in range(num_pop):
                res, nov=expectation(x, l=p_i, k=k_val, G=50, space_boundaries=[0, high_b])
                results.append(res)
                novs.append(nov)
            #plt.plot(results,"b-");plt.show()

            flag=any([x<1e-8 for x in results])
            has_zero.append(flag)
            sol=np.argmin(results)
            ineq=[x<k_val*y for x,y in zip(results,novs)]
            print("(Expectation) min, argmin==", np.min(results), sol)
            print("(Novelty) min, argmin==", np.min(novs), np.argmin(novs))
            print("************* ineq*****************")
            print(ineq)
            plt.plot(x[1,:],x[0,:],"ro");
            plt.plot(x[1,sol],x[0,sol],"bo")
            plt.plot(x[1,np.argmin(novs)],x[0,np.argmin(novs)],"yx")
            plt.show()
        
        #plt.plot([int(x) for x in has_zero],"b-");plt.show()


         

