import math
import numpy as np
from copy import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

import dreal.symbolic as symbolic
from dreal.api import CheckSatisfiability, Minimize


class LPO:
    def __init__(self, hidden_size, num_inputs, action_space, policy):
        self.action_space = action_space
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        #self.model = SingleLayerPolicy(hidden_size, num_inputs, action_space)
        self.model = policy
        self.model.train()

        weights, limits = self.model.generate_variable_limits(hidden_size, num_inputs, action_space, [-1,1])

        self.vars = weights
        self.limits = limits
        self.constraints = []
        self.training_samples = {"states": np.array([]).reshape(0,4),
                                  "discounted_rewards":np.array([]), 
                                  "actions":np.array([]).astype(int)}

    def select_action(self, state):
        probs = self.model(Variable(state))   
        action = probs.multinomial(len(probs)).data
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()
        return action[0], log_prob, entropy

    def update_parameters(self, states, actions, rewards, log_probs, entropies, gamma):

        discounted_rewards = [0]
        for r in rewards[::-1]:
            discounted_rewards = [r + gamma*discounted_rewards[0]] + discounted_rewards

        rounded_states = np.around(states, 2) #to be experimented

        constraints = self.constraints 

        for i, i_state in enumerate(rounded_states[:-1]):
            for j, j_state in enumerate(rounded_states[i+1:]):
                if np.all(np.equal(i_state, j_state)) and (actions[i] != actions[j]):
                    #need to explicitly formulate these things 
                    print "constraint! state: " 
                    print i_state
                    prob = self.model.forward_symbolic(self.vars, i_state)
                    c = ((prob[actions[i]] - prob[actions[j]]) * (discounted_rewards[i] - discounted_rewards[j]) > 0)
                    constraints.append(c)
            for j, j_state in enumerate(self.training_samples["states"]):
                if np.all(np.equal(i_state, j_state)) and (actions[i] != self.training_samples["actions"][j]):
                    #need to explicitly formulate these things 
                    print "constraint! state: " 
                    print i_state
                    prob = self.model.forward_symbolic(self.vars, i_state)
                    c = ((prob[actions[i]] - prob[self.training_samples["actions"][j]]) * \
                     (discounted_rewards[i] - self.training_samples["discounted_rewards"][j]) > 0)
                    constraints.append(c)


        if len(constraints) < 1:
            #If there are not enough traing samples, we postpone training to next iteration
            self.constraints = constraints
            self.training_samples["states"] = np.concatenate([self.training_samples["states"],rounded_states])
            self.training_samples["discounted_rewards"] = np.concatenate([self.training_samples["discounted_rewards"], discounted_rewards])
            # hack. if the sequence terminates, there will be 1 more state than action
            # we pad the action sequence such that their indices align 
            actions_padding = [-1]*(len(rounded_states)-len(actions))
            self.training_samples["actions"] = np.concatenate([self.training_samples["actions"],actions,actions_padding])
            assert(len(self.training_samples["states"]) == len(self.training_samples["discounted_rewards"]))
            assert(len(self.training_samples["states"]) == len(self.training_samples["actions"]))

        else:
            print("# of Constraints: %d "%len(constraints))
            constraints += self.limits 
            #print constraints
            print("optimizer starts")
            #print(constraints)
            #print constraints
            f_sat = symbolic.logical_and(*constraints)
            timer_start = time.time()
            result = CheckSatisfiability(f_sat, 1)
            if not result:
                #what happens if not satisfiable?
                print("not satisfiable, move on")
            else:
                #update weights
                updated_weights = self.get_weights_from_result(result)
                self.model.update_weights(updated_weights)
                print("weights updated")

            timer_end = time.time()
            print("Solver timer: %.2f s" % (timer_end - timer_start))

            #re-initialize
            self.constraints = []
            self.training_samples = {"states": np.array([]).reshape(0,4), 
                                    "discounted_rewards":np.array([]), 
                                    "actions":np.array([]).astype(int)}
    
    def get_weights_from_result(self, result, print_=True):
        if print_:
            print result

        layers = self.model.children()
        updated = []
        for k,l in enumerate(layers):

            if print_:
                print l.weight.data #old weights

            l_weights = l.weight.data.tolist()
            for i in range(l.weight.data.size(0)):
                for j in range(l.weight.data.size(1)):
                    index = result.index(self.vars[k][i][j])
                    lb = result[index].lb()
                    ub = result[index].ub()
                    if l_weights[i][j] < lb or l_weights[i][j] > ub:
                        l_weights[i][j] = ((lb+ub)/2)
                        #update only if current weight is not in the solved lower/upper bound
            updated.append(torch.Tensor(l_weights))
            if print_:
                print updated[-1]

        return updated