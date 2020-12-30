
import gym
import matplotlib.pyplot as plt
import numpy as np




class ArchimedeanSpiral(gym.Env):

    def __init__(self, a=0.1, limit=10):

        self.a=a


        #self.phi_vals=np.linspace(-10*np.pi, 10*np.pi, 1000)
        self.phi_vals=np.linspace(0, limit*np.pi, 10000)

        self.r_vals=[self.a * x for x in self.phi_vals]

        self.x_vals=[self.a * x * np.cos(x) for x in self.phi_vals]
        self.y_vals=[self.a * x * np.sin(x) for x in self.phi_vals]

        self.start=[0,0]
        self.goal=[self.x_vals[-1], self.y_vals[-1]]

    def compute_value(self, phi):

        return [self.a * phi * np.cos(phi), self.a * phi * np.sin(phi)]


    def render(self,hold_on=False):
        plt.plot(self.x_vals, self.y_vals,color="black")
        plt.plot(self.start[0], self.start[1],color="red",marker="o")
        plt.plot(self.goal[0], self.goal[1],color="green",marker="o")
        plt.axis("equal")


    def step(self,action):
        return 0


    def reset(self):
        return 0


    def close(self):
        return 0


if __name__=="__main__":

    spiral=ArchimedeanSpiral()
    spiral.render()
    plt.show()
