import torch
import torch.nn as nn
import torch.nn.functional as F
import dreal.symbolic as symbolic

class TwoLayerPolicy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(TwoLayerPolicy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.n
        self.linear1 = nn.Linear(num_inputs, hidden_size,bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size,bias=False)
        self.linear3 = nn.Linear(hidden_size, num_outputs,bias=False)

    def forward(self, inputs):
        x = inputs
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        action_scores = self.linear3(x)
        return F.softmax(action_scores)

    def forward_symbolic(self, var, inputs):
        x = inputs
        layer1_weights, layer2_weights, layer3_weights = var

        layer1 = []
        for i in range(self.linear1.weight.size(0)):
            ex = 0
            for j in range(self.linear1.weight.size(1)):
                ex += inputs[j]*layer1_weights[i][j]
            layer1.append(symbolic.tanh(ex))

        layer2 = []
        for i in range(self.linear2.weight.size(0)):
            ex = 0
            for j in range(self.linear2.weight.size(1)):
                ex += layer1[j] * layer2_weights[i][j]
            layer2.append(symbolic.tanh(ex))

        layer3 = []
        for i in range(self.linear3.weight.size(0)):
            ex = 0
            for j in range(self.linear3.weight.size(1)):
                ex += layer2[j]*layer3_weights[i][j]
            layer3.append(ex)

        softmax = [ symbolic.exp(layer3[0])/(symbolic.exp(layer3[0]) + symbolic.exp(layer3[1])),
                    symbolic.exp(layer3[1])/(symbolic.exp(layer3[0]) + symbolic.exp(layer3[1]))] 
        return softmax

    def generate_variable_limits(self, hidden_size, num_inputs, action_space, bounds):

        layer1_weights = [[0 for i in range(num_inputs)] for j in range(hidden_size)] #placeholder
        for i in range(hidden_size):
            for j in range(num_inputs):
                layer1_weights[i][j] = symbolic.Variable("A" + str(i) + "_" + str(j))
        
        layer2_weights = [[0 for i in range(hidden_size)] for j in range(hidden_size)]
        for i in range(hidden_size):
            for j in range(hidden_size):
                layer2_weights[i][j] = symbolic.Variable("B" + str(i) + "_" + str(j))

        layer3_weights = [[0 for i in range(hidden_size)] for j in range(action_space.n)]
        for i in range(action_space.n):
            for j in range(hidden_size):
                layer3_weights[i][j] = symbolic.Variable("C" + str(i) + "_" + str(j))

        limits = []
        var_lb, var_ub = bounds[0], bounds[1]
        for l in layer1_weights+layer2_weights+layer3_weights:
            for p in l:
                limits.append((p <= var_ub))
                limits.append((p >= var_lb))
       
        return (layer1_weights, layer2_weights, layer3_weights), limits
        
    def update_weights(self, updated_w):
        self.linear1.weight.data, self.linear2.weight.data, self.linear3.weight.data = updated_w
