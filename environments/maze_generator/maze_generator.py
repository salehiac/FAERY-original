import matplotlib.pyplot as plt
import numpy as np
import pdb
import cv2

class Maze:
    """
    Class generating random mazes that can be saved as *pbm for use with libfastsim.
    """

    def __init__(self, N, B_s=16, W_s=3, outer_w=5):
        """
        N        the maze will be based on a grid of shape NxN
        B_s      size of cells in pixels
        W_s      size of walls in pixels
        outer_w  thickness of image borders (outer wall)
        """

        #height/widht without the external border
        self.height=N*B_s + (N-1)*W_s
        self.width=self.height
        
        self.mz=np.ones([self.height, self.width])

        self.B_s=B_s
        self.W_s=W_s
        self.outer_w=outer_w
        self.N=N

        self.num_saved=0

    def __len__(self):
        """
        for compatibility. While not strictly True, "Maze" will be considered as a dataset with inifinite samples
        """
        inf=10000000
        return inf

    def save(self,out_d="/tmp/",for_use_with_libfastsim=True):
        """
        libfastsim requires size that is divisible by 8
        returns file_name
        """

        fn=out_d+"/maze_"+str(self.num_saved)+".pbm"
        if for_use_with_libfastsim:
            cv2.imwrite(fn,cv2.resize(self.mz,(200,200)))
        else:
            cv2.imwrite(fn,self.mz)
        self.num_saved+=1

        return fn


    def generate(self):

        self.clear()
        self.enable_all_walls_and_fill()

        visited=np.zeros([self.N, self.N])

        r_i=np.random.randint(self.N)
        r_j=np.random.randint(self.N)

        self.dfs(r_i, r_j, visited)

        #add external border
        wall_h=np.zeros([self.outer_w, self.width])
        self.mz=np.concatenate([wall_h, self.mz, wall_h],0)
        wall_v=np.zeros([self.height + 2*self.outer_w, self.outer_w])
        self.mz=np.concatenate([wall_v, self.mz, wall_v],1)


    def dfs(self, i, j, visited):

        visited[i,j]=1

        neis=[]
        if i>0:
            neis.append((i-1, j))
        if j>0:
            neis.append((i, j-1))
        if i<self.N-1:
            neis.append((i+1,j))
        if j<self.N-1:
            neis.append((i,j+1))

        neis_p=np.random.permutation(neis)

        for nb in neis_p:

            if not visited[nb[0], nb[1]]:
                self.set_wall_between_two_cells(val=1,
                        a_x=i,
                        a_y=j,
                        b_x=nb[0],
                        b_y=nb[1],
                        fill=False)
                self.dfs(nb[0], nb[1], visited)

    def enable_all_walls_and_fill(self):

        for i in range(self.N):#iterating over the diagonal
            for j in range(1,self.N):
                #fill is set to True because otherwise the corners between the walls remains empty
                self.set_wall_between_two_cells(0, i, j-1, i, j, fill=True)#vertical walls
                self.set_wall_between_two_cells(0, j-1, i, j, i, fill=True)#horizontal walls

    def clear(self):
        self.mz=np.ones([self.height, self.width])


    def set_wall_between_two_cells(self, val, a_x, a_y, b_x, b_y,fill=False):
        """
        x,y respectively vertical, horizontal

        fill    boolean, if True then the corners between walls is filled
        """

        d_x=np.abs(a_x-b_x)
        d_y=np.abs(a_y-b_y)

        if d_x==1 and d_y==0:
            m_x=min(a_x, b_x)
           
            s_x=self.B_s*(m_x+1) + self.W_s*m_x
            wall_lims_x=[s_x, s_x+self.W_s]
            wall_lims_y=[b_y*self.B_s+b_y*self.W_s, (b_y+1)*self.B_s+b_y*self.W_s]

            self.mz[wall_lims_x[0]:wall_lims_x[1], wall_lims_y[0]:wall_lims_y[1]]=val

            if fill and b_y!=self.N-1:
                self.mz[wall_lims_x[0]:wall_lims_x[1], wall_lims_y[1]:wall_lims_y[1]+self.W_s]=val
        
        if d_y==1 and d_x==0:
            
            m_y=min(a_y, b_y)
           
            s_y=self.B_s*(m_y+1) + self.W_s*m_y
            wall_lims_y=[s_y, s_y+self.W_s]
            wall_lims_x=[b_x*self.B_s+b_x*self.W_s, (b_x+1)*self.B_s+b_x*self.W_s]

            self.mz[wall_lims_x[0]:wall_lims_x[1], wall_lims_y[0]:wall_lims_y[1]]=val
            
            if fill and b_x!=self.N-1:
                self.mz[wall_lims_x[1]:wall_lims_x[1]+self.W_s, wall_lims_y[0]:wall_lims_y[1]]=val

    def show(self):

        plt.imshow(self.mz)
        plt.show()



if __name__=="__main__":

    num_mazes=600
    maze=Maze(10)
    for i in range(num_mazes):
        maze.generate()
        #maze.show()
        maze.save("/tmp/mazes_10x10_train/")
        
    for i in range(100):
        maze.generate()
        #maze.show()
        maze.save("/tmp/mazes_10x10_test/")
