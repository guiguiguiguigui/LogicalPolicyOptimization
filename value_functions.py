# A simple neural net value function that serves as a baseline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class nnValueFunction(nn.Module):
    # Here number of epocks and stepsizes are arbitrarily chosen
    def __init__(self, ob_dim=None, n_epochs=20, stepsize=1e-3):
        super(nnValueFunction, self).__init__()
       
        self.ob_dim = ob_dim
        self.n_epochs = n_epochs
        self.stepsize = stepsize

        self.linear1 = nn.Linear(ob_dim, 50)
        self.linear2 = nn.Linear(50, 50)
        self.pred = nn.Linear(50,1)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=stepsize, momentum=0.9)
    
    def forward(self, inputs):
        x = inputs
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        pred = self.pred(x)
        y = pred.view(-1) #equivalent of reshape, (?,1) --> (?,)
        return y
    
    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        for epoch in range(self.n_epochs):
            running_loss = 0.0

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.forward(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

