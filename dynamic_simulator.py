#!/usr/bin/env python2

from matplotlib import pyplot as plt

class SpringMass():

    def __init__(self):

        self.states = 1
        self.actions = 1

        self.dt = 1.0
        self.pos = 0.0
        self.spd = 0.0
        self.acc = 0.0
        self.mass = 2.0
        self.spring_k = 0.5
        self.spring_l = 10.0
        self.damper_c = 0.0


    def update(self,force):

        spring_force = (self.spring_l - self.pos) * self.spring_k
        damper_force = - self.damper_c * self.spd ** 2
        self.acc = (spring_force + damper_force + force) / self.mass

        self.spd = self.spd + self.acc * self.dt
        self.pos = self.pos + self.spd * self.dt

        # print(self.pos,self.spd,self.acc)

        return self.pos


if __name__ == "__main__":
    print("Hello There")
    sm = SpringMass()

    pos_array = []
    for i in range(100):
        pos_array.append(sm.update(0.0))

    plt.plot(pos_array)
    plt.show()
