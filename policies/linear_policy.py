
import numpy as np
from scipy.special import expit
import dreal

class LinearPolicy(object):
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space #output dimention
        num_outputs = action_space.n
        self.num_inputs = num_inputs
        self.coef = [0 for _ in (range(num_inputs)+1)]
    
    def forward(self, inputs):
        x = np.array(inputs)
        y = np.dot(x, self.coef[:-1]) + self.coef[:-1] 
        return expit(y) #sigmoid fuction

    def forward_symbolic(self, var, inputs):
        x = inputs
        var_coef = var
        y = [x[i] * v for v in var_coef[:-1]] + var_coef[-1]
        sigmoid = 1 / (1 + dreal.exp(-y))
        return sigmoid

    def generate_variable_limits(self, hidden_size, num_inputs, action_space, bounds):
        lb, ub = bounds
        coef_vars = []
        limits = []
        for i in range(hidden_size):
            v = dreal.Variable(str(i))
            var_coefs.append(v)
            limits.append((v <= ub))
            limits.append((v >= lb))
        return coef_vars, limits

    def update_weights(self, updated_w):
        self.coef = updated_w

