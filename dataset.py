#!/usr/bin/env python2
from dynamic_simulator import SpringMass
import random
import torch
import math
from matplotlib import pyplot as plt
class DynamicDataset():
    def __init__(self,length,seq_len):
        self.length = length
        self.seq_len = seq_len

    def __len__(self):
        return self.length

    def __getitem__(self,index):
        sm = SpringMass()
        actions = torch.zeros([self.seq_len, sm.actions])
        states = torch.zeros([self.seq_len, sm.states])
        next_states = torch.zeros([self.seq_len, sm.states])

        sm.pos = random.uniform(8.0,12.0)
        sm.spd = random.uniform(-1.0,1.0)


        wave_len = random.uniform(40,60)
        phase = random.uniform(-math.pi, math.pi)
        mag = random.uniform(0.5,1.0)

        for i in range(self.seq_len):
            force = math.sin(float(i)/wave_len*2*math.pi + phase)*mag

            states[i,0] = sm.pos
            actions[i,0] = force
            next_states[i,0] = sm.update(force)

        return {"next_states": next_states, "states": states, "actions":actions}

if __name__ == "__main__":
    print("Hello There")
    d = DynamicDataset(1,100)

    for i in range(5):
        sample = d[i]
        next_states = sample["next_states"].numpy()
        plt.plot(next_states[:,0])
    plt.show()
