import torch
import torch.nn as nn
import torch.nn.functional as F
import dreal

class SingleLayerPolicy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(SingleLayerPolicy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.n
        self.linear1 = nn.Linear(num_inputs, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, num_outputs, bias=False)

    def forward(self, inputs):
        x = inputs
        x = torch.tanh(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores)

    def forward_symbolic(self, var, inputs):
        x = inputs
        layer1_weights, layer2_weights = var
        layer1 = []
        for i in range(self.linear1.weight.size(0)):
            #in the case where wer have bias. Currently disabled
            #ex = self.linear1.bias.data[i].item() 
            ex = 0
            for j in range(self.linear1.weight.size(1)):
                ex += inputs[j]*layer1_weights[i][j]
            layer1.append(dreal.tanh(ex))

        layer2 = []
        for i in range(self.linear2.weight.size(0)):
            #ex = self.linear2.bias.data[i].item()
            ex = 0
            for j in range(self.linear2.weight.size(1)):
                ex += layer1[j] * layer2_weights[i][j]
            layer2.append(ex)

        softmax = [ dreal.exp(layer2[0])/(dreal.exp(layer2[0]) + dreal.exp(layer2[1])),
                    dreal.exp(layer2[1])/(dreal.exp(layer2[0]) + dreal.exp(layer2[1]))] 
        return softmax

    def generate_variable_limits(self, hidden_size, num_inputs, action_space, bounds):
        var_lb, var_ub = bounds
        layer1_weights = [[0 for i in range(num_inputs)] for j in range(hidden_size)] #placeholder
        for i in range(hidden_size):
            for j in range(num_inputs):
                layer1_weights[i][j] = dreal.Variable("A" + str(i) + "_" + str(j))
        
        layer2_weights = [[0 for i in range(hidden_size)] for j in range(action_space.n)]
        for i in range(action_space.n):
            for j in range(hidden_size):
                layer2_weights[i][j] = dreal.Variable("B" + str(i) + "_" + str(j))

        limits = [] #does this change performance?
        
        for l in layer1_weights:
            for p in l:
                limits.append((p <= var_ub))
                limits.append((p >= var_lb))
        for l in layer2_weights:
            for p in l:
                limits.append((p <= var_ub))
                limits.append((p >= var_lb))
        return (layer1_weights, layer2_weights), limits
    
    def update_weights(self, updated_w):
        self.linear1.weight.data, self.linear2.weight.data = updated_w
