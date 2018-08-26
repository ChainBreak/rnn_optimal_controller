#!/usr/bin/env python2
import torch
import torch.nn as nn
from dataset import DynamicDataset
from torch.utils.data import DataLoader
from torch import optim
from matplotlib import pyplot as plt

class DynamicModel(nn.Module):
    def __init__(self,state_size, action_size, hidden_size):
        super(DynamicModel,self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(state_size + action_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, state_size)

    def forward(self,state,action,hidden):

        combined = torch.cat((state,action,hidden),dim=-1)
        # print(combined.shape)
        hidden = torch.clamp(self.i2h(combined),-1000,1000)

        next_state = self.h2o(hidden)
        return hidden, next_state




if __name__ == "__main__":
    print("Hello There")

    dynamic_dataset = DynamicDataset(6000,seq_len=50)
    batch_size = 20
    train_loader = DataLoader(dynamic_dataset, batch_size=batch_size, num_workers=4)

    n_hidden = 5
    model = DynamicModel(1,1,n_hidden)
    model.train()


    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.00001)#, momentum=0.9)

    #for each batch
    for i_batch, sample_batch in enumerate(train_loader):
        states = sample_batch["states"]
        next_states = sample_batch["next_states"]
        actions = sample_batch["actions"]

        hidden = torch.zeros((batch_size,n_hidden))
        model.zero_grad()

        loss_sum = torch.zeros(1)

        seq_len = states.shape[1]
        #unroll sequence
        for i in range(seq_len):
            state = states[:,i,:]
            next_state = next_states[:,i,:]
            action = actions[:,i,:]

            if i < 100:
                hidden,estimated_next_state = model(state,action,hidden)
            else:
                hidden,estimated_next_state = model(estimated_next_state,action,hidden)

            if i > seq_len/2 or True:
                loss = criterion(estimated_next_state,next_state)
                # print(estimated_next_state,next_state)
                # print(loss)
                loss_sum += loss

        loss_sum /= i/2
        loss_sum.backward()
        print("Loss: " + str(loss_sum))
        # for p in model.parameters():
        #     print(p.grad.data)
        torch.nn.utils.clip_grad_norm_(model.parameters(),500)
        # for p in model.parameters():
        #     print(p.grad.data)
        optimizer.step()
        # raw_input("sadf")

    # test_out = []
    # for i in range(1000):
    #     hidden,estimated_next_state = model(estimated_next_state,action,hidden)
    #     test_out.append(estimated_next_state.data.numpy())
    # plt.plot(test_out)
    # plt.show()

    print(model.i2h.weight)
    print(model.i2h.bias)

    hidden = torch.zeros((n_hidden))
    loss_array = []
    out_array = []
    for i in range(seq_len):
        state = states[0,i,:]
        next_state = next_states[0,i,:]
        action = actions[0,i,:]
        # print(state)
        if i < seq_len/2:
            hidden,estimated_next_state = model(state,action,hidden)
        else:
            hidden,estimated_next_state = model(estimated_next_state,action,hidden)
        out_array.append(estimated_next_state.data.numpy())

        loss = criterion(estimated_next_state,next_state)
        loss_array.append(loss.data.numpy())
    plt.plot(out_array)
    plt.plot(states[0,:,0].numpy())
    plt.plot(actions[0,:,0].numpy())

    plt.show()
