import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pdb
import pickle

def read_tables_and_meta_agents(dir_name):

    fns=os.listdir(dir_name)

    train_table_fs=sorted([dir_name+"/"+x for x in fns if "evolution_table_train" in x])
    test_table_fs=sorted([dir_name+"/"+x for x in fns if "evolution_table_test" in x])
    pop_fs=sorted([dir_name+"/"+x for x in fns if "population_prior" in x ])

    train_tables=[]
    for fn in train_table_fs:
        print("loading ",fn)
        x_fn=np.load(fn)
        train_tables.append(x_fn["arr_0"])

    print("=============================================== loaded train_tables")

    test_tables=[]
    for fn in test_table_fs:
        print("loading ",fn)
        x_fn=np.load(fn)
        test_tables.append(x_fn["arr_0"])

    print("=============================================== loaded test_tables")

    agents=[]
    for fn in pop_fs:
        print("loading ",fn)
        with open(fn,"rb") as fl:
            agents.append(pickle.load(fl))
        

    print("=============================================== loaded agents")

    return train_tables, test_tables, agents

def plot_tables(tb_list,msg):
    for i in range(len(tb_list)):
        plt.imshow(tb_list[i]);
        plt.title(f"table {i} "+msg )
        plt.show()

if __name__=="__main__":

    t_train, t_test, agents = read_tables_and_meta_agents(sys.argv[1])

    plot_tables(t_train, "(train)")
    plot_tables(t_test, "(test)")
