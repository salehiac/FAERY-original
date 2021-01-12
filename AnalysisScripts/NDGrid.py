import numpy as np
import matplotlib.pyplot as plt

class NDGridUniformNoPrealloc:

    def __init__(self, Gs, dims, lower_bounds, higher_bounds):
        """
        Gs is a list and is the number of grids per dimension
        lower_bounds and higher_bounds should both be lists of length dims, giving the bounds for each dimension
        """

        self.Gs=Gs
        self.dims=dims
        self.num_cells=1
        for x in Gs:
            self.num_cells*=x
        self.lb=lower_bounds
        self.hb=higher_bounds

        self.ranges=[]
        for i in range(dims):
            self.ranges.append(list(np.linspace(self.lb[i], self.hb[i], self.Gs[i]+1)))

        self.visited_cells={}

    def compute_current_coverage(self):

        return len(self.visited_cells)/self.num_cells

    def visit_cell(self,pt):
        """
        pt should be of size 1*self.dims
        """

        cell=[]
        for i in range(self.dims):
            z=pt[i]
            if z>self.hb[i] or z<self.lb[i]:
                raise Exception("input point is outside of grid boundaries")

            for ii in range(1,self.Gs[i]+1):
                if self.ranges[i][ii]>=z:
                    cell.append(ii-1)
                    break

        assert len(cell)==self.dims, "this should'nt happen"

        cell=tuple(cell)
        print("cell==\n",cell)

        if cell not in self.visited_cells:
            self.visited_cells[cell]=1
        else:
            self.visited_cells[cell]+=1







if __name__=="__main__":

    if 0:
        grid1d=NDGridUniformNoPrealloc(10, 1, -5,5)
        grid1d.visit_cell([-3.6])
        grid1d.visit_cell([-4.3])
        grid1d.visit_cell([4.2])

        print(grid1d.visited_cells)
        print(grid1d.num_cells)
        print(grid1d.range)
        
    if 1:
        grid2d=NDGridUniformNoPrealloc(6, 2, -10,5)
        grid2d.visit_cell([-3.6, 2.4])

        print(grid2d.visited_cells)
        print(grid2d.num_cells)
        print(grid2d.range)








