#!/usr/bin/env python2

from matplotlib import pyplot as plt
import numpy as np

class SpringMass():

    def __init__(self):

        self.states = 1
        self.actions = 1

        pos = 2.0
        spd = 0.0
        acc = 0.0
        self.x = np.array([[pos,spd]]).T

        dt = 0.1
        mass = 1.0
        spring_k = 20.0
        damper_c = 2.0
        self.A = np.array([[0,1],[-spring_k/mass, -damper_c/mass]])
        self.B = np.array([[0,5.0/mass]]).T
        # print(self.A)
        self.A = np.eye(2) + self.A*dt
        self.B = self.B*dt
        #self.A = np.exp(self.A*dt)



        # print(self.A)

    def set_x(self,pos,spd):
        self.x = np.array([[pos,spd]]).T


    def update(self,force):
        # self.x[2] += force
        self.x = np.matmul(self.A,self.x) + self.B*force
        # print(self.x)

        return float(self.x[0])


if __name__ == "__main__":
    print("Hello There")
    sm = SpringMass()

    pos_array = []
    for i in range(100):
        pos_array.append(sm.update(100.0))

    plt.plot(pos_array)
    plt.show()
