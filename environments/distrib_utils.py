import numpy as np
import scipy.special


def KLdiv(P,Q):
    """
    P, Q 2d distributions
    """
    C=np.log(P/Q)
    return (P*C).sum()

def jensen_shannon(P,Q):
    M=0.5*(P+Q)

    return 0.5*KLdiv(P,M) + 0.5*KLdiv(Q,M)

def uniform_like(A):

    u=np.ones_like(A)
    return scipy.special.softmax(u)




if __name__=="__main__":
    

    A=np.random.rand(10,10);
    B=np.random.rand(10,10);
    A=scipy.special.softmax(A)
    B=scipy.special.softmax(B)

    kl_AA=KLdiv(A,A)
    kl_BB=KLdiv(B,B)
    kl_AB=KLdiv(A,B)
    kl_BA=KLdiv(B,A)

    print("kl_AA=",kl_AA)
    print("kl_BB=",kl_BB)
    print("kl_AB=",kl_AB)
    print("kl_BA=",kl_BA)

    js_AA=jensen_shannon(A,A)
    js_BB=jensen_shannon(B,B)
    js_AB=jensen_shannon(A,B)
    js_BA=jensen_shannon(B,A)

    print("js_AA=",js_AA)
    print("js_BB=",js_BB)
    print("js_AB=",js_AB)
    print("js_BA=",js_BA)

    U=uniform_like(A)
    print(U)

    




