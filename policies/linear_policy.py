
import numpy as np
from scipy.special import expit
import dreal
import math

class LinearPolicy(object):
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space #output dimention
        num_outputs = action_space.n
        self.num_inputs = num_inputs
        self.coef = np.random.rand(num_inputs)
    
    def forward(self, inputs):
        x = inputs[0]
        y = np.dot(x, self.coef) + 1 #offset
        y_prob = 1 / (1 + math.exp(-y)) #sigmoid fuction
        probs = [1-y_prob, y_prob]  
        return probs

    def forward_symbolic(self, var, inputs):
        x = inputs
        var_coef = var
        y = var_coef[-1]
        for i in range(len(inputs)):
            y += x[i] * var_coef[i] 
        sigmoid = 1 / (1 + dreal.exp(-y))
        return [1-sigmoid, sigmoid]
    
    def forward_symbolic_y(self, var, inputs):
        x = inputs
        y = 1
        for i in range(len(inputs)):
            y += x[i] * var[i] 
        return y


    def generate_variable_limits(self, hidden_size, num_inputs, action_space, bounds):
        lb, ub = bounds
        coef_vars = []
        limits = []
        for i in range(num_inputs):
            v = dreal.Variable("coef_"+str(i))
            coef_vars.append(v)
            limits.append((v <= ub))
            limits.append((v >= lb))
        return coef_vars, limits

    def update_weights(self, updated_w):
        self.coef = updated_w

    def __call__(self,inputs):
        return self.forward(inputs)