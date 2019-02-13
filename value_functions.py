# value functions that serve as baseline
# a linear one and a nn one

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class linearValueFunction(object):
    """ Estimates the baseline function for PGs via ridge regression. """

    def __init__(self):
        self.coef = None

    def fit(self, X, y):
        """ 
        Updates weights (self.coef) with design matrix X (i.e. observations) and
        targets (i.e. actual returns) y. 
        """
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)

    def predict(self, X):
        """ Predicts return from observations (i.e. environment states) X. """
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)

    def preproc(self, X):
        """ Adding a bias column, and also adding squared values (sketchy?). """
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)


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

    def predict(self, inputs):
        return self.forward(torch.Tensor(inputs))
    
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

