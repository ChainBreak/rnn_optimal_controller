#!/usr/bin/env python2
import torch
import torch.nn as nn
from dataset import DynamicDataset
from torch.utils.data import DataLoader
from torch import optim
from matplotlib import pyplot as plt

class DynamicModel(nn.Module):
    def __init__(self,x_size, u_size, y_size):
        """
            x_size system hidden state
            u_size system inputs
            y_size system output or measurement
        """
        super(DynamicModel,self).__init__()
        self.x_size = x_size

        self.predict_y = nn.Linear(x_size + u_size, y_size, bias=False)
        self.predict_x = nn.Linear(x_size + u_size + y_size, x_size, bias=False)

        self.x = None

    def forward(self,u,y,use_error):
        if self.x is None:
            if len(u.shape)>1:
                batch_size = u.shape[0]
                self.x = torch.zeros((batch_size,self.x_size))
            else:
                self.x = torch.zeros(self.x_size)


        y_hat = self.predict_y( torch.cat((self.x,u),dim=-1))
        y_hat = torch.clamp(y_hat,-1000,1000)

        if use_error:
            error = y_hat - y
        else:
            error = torch.zeros(y.shape)
        self.x = self.predict_x( torch.cat((self.x,u,error),dim=-1))
        self.x = torch.clamp(self.x,-1000,1000)

        return y_hat

    def zero_x(self):
        self.x = None




if __name__ == "__main__":
    print("Hello There")

    dynamic_dataset = DynamicDataset(10000,seq_len=500)
    batch_size = 20
    train_loader = DataLoader(dynamic_dataset, batch_size=batch_size, num_workers=4)

    model = DynamicModel(x_size = 5, u_size = 1, y_size = 1)
    model.train()


    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

    #for each batch
    for i_batch, sample_batch in enumerate(train_loader):
        Y = sample_batch["states"]
        U = sample_batch["actions"]

        model.zero_grad()
        model.zero_x()
        loss_sum = torch.zeros(1)

        seq_len = Y.shape[1]
        #unroll sequence
        for i in range(seq_len):
            y = Y[:,i,:]
            u = U[:,i,:]

            y_hat = model(u,y,use_error=True)

            # if i < 100:
            #     hidden,estimated_next_state = model(state,action,hidden)
            # else:
            #     hidden,estimated_next_state = model(estimated_next_state,action,hidden)

            # if i > seq_len/2 or True:
            loss = criterion(y_hat,y)
            # print(estimated_next_state,next_state)
            # print(loss)
            loss_sum += loss

        loss_sum /= i
        loss_sum.backward()
        print("Loss: %f" % float(loss_sum.data.numpy()))
        # for p in model.parameters():
        #     print(p.grad.data)
        torch.nn.utils.clip_grad_norm_(model.parameters(),500)
        # for p in model.parameters():
        #     print(p.grad.data)
        optimizer.step()
        # raw_input("press any key\n")

    # test_out = []
    # for i in range(1000):
    #     hidden,estimated_next_state = model(estimated_next_state,action,hidden)
    #     test_out.append(estimated_next_state.data.numpy())
    # plt.plot(test_out)
    # plt.show()

    print(model.predict_x.weight)
    print(model.predict_x.bias)


    loss_array = []
    out_array = []
    model.zero_x()
    for i in range(seq_len):
        y = Y[0,i,:]
        u = U[0,i,:]

        if i < seq_len/2:
            y_hat = model(u,y,use_error=True)
        else:
            y_hat = model(u,y,use_error=False)
        out_array.append(y_hat.data.numpy())

        loss = criterion(y_hat,y)
        loss_array.append(loss.data.numpy())
    plt.plot(out_array)
    plt.plot(Y[0,:,0].numpy())
    plt.plot(U[0,:,0].numpy())

    plt.show()
