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

import dreal

class LPO:
    def __init__(self, hidden_size, num_inputs, action_space, constraint_size, policy):
        self.action_space = action_space
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.constraint_size = constraint_size
        self.model = policy

        self.isLinear = False
        try:
            self.model.train()
        except AttributeError:
            self.isLinear = True
            print("linear policy here")
            pass #linear policy
        weights, limits = self.model.generate_variable_limits(hidden_size, num_inputs, action_space, [-50,50])

        self.vars = weights
        self.limits = limits
        self.constraints = []
        self.training_samples = {"states": np.array([]).reshape(0,4),
                                  "advantage":np.array([]), 
                                  "actions":np.array([]).astype(int)}

    def select_action(self, state):
        probs = self.model(Variable(state))   
        if self.isLinear:
            # TODO: this only works in th cartpole case
            action = np.where(np.random.multinomial(1,probs)==1)[0][0]
            action = int(action)
            # TODO: implement
            log_prob, entropy = 0,0
        else:
            actions = probs.multinomial(len(probs)).data
            action = actions[0][0].numpy()
            prob = probs[:, actions[0,0]].view(1, -1)
            log_prob = prob.log()
            entropy = - (probs*probs.log()).sum()

        return action, log_prob, entropy

    def update_parameters(self, states, actions, advantage):

        rounded_states = np.around(states, 2) #to be experimented with

        constraints = self.constraints 

        for i, i_state in enumerate(rounded_states[:-1]):
            for j, j_state in enumerate(rounded_states[i+1:]):
                if np.all(np.equal(i_state, j_state)) and (actions[i] != actions[j]):
                    #need to explicitly formulate these things 
                    print "constraint %d! state: " % (len(constraints)+1)
                    print i_state
                    prob = self.model.forward_symbolic(self.vars, i_state)
                    c = ((prob[actions[i]] - prob[actions[j]]) * (advantage[i] - advantage[j]) > 0)
                    constraints.append(c)
            for j, j_state in enumerate(self.training_samples["states"]):
                if np.all(np.equal(i_state, j_state)) and (actions[i] != self.training_samples["actions"][j]):
                    #need to explicitly formulate these things 
                    print "constraint %d! state: " % (len(constraints)+1)
                    print i_state
                    prob = self.model.forward_symbolic(self.vars, i_state)
                    c = ((prob[actions[i]] - prob[self.training_samples["actions"][j]]) * \
                     (advantage[i] - self.training_samples["advantage"][j]) > 0)
                    constraints.append(c)

        if len(constraints) < self.constraint_size:
            #If there are not enough traing samples, we postpone training to next iteration
            self.constraints = constraints
            self.training_samples["states"] = np.concatenate([self.training_samples["states"],rounded_states])
            self.training_samples["advantage"] = np.concatenate([self.training_samples["advantage"], advantage])
            # hack. if the sequence terminates, there will be 1 more state than action
            # we pad the action sequence such that their indices align 
            actions_padding = [-1]*(len(rounded_states)-len(actions))
            self.training_samples["actions"] = np.concatenate([self.training_samples["actions"],actions,actions_padding])
            assert(len(self.training_samples["states"]) == len(self.training_samples["advantage"]))
            assert(len(self.training_samples["states"]) == len(self.training_samples["actions"]))

            return False

        else:
            print("# of Constraints: %d "%len(constraints))
            constraints += self.limits 
            #print constraints
            print("optimizer starts")
            #print(constraints)
            #print constraints
            f_sat = dreal.logical_and(*constraints)
            timer_start = time.time()
            result = dreal.CheckSatisfiability(f_sat, 0.01)
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
                                    "advantage":np.array([]), 
                                    "actions":np.array([]).astype(int)}
            return True
    
    def get_weights_from_result(self, result, print_=True):
        if print_:
            print "Check_Satisfiability result:"
            print result

        if self.isLinear:
            l = self.model.coef
            if print_:
                print "Old coefficients:"
                print l #old weights
            updated = []
            for i in range(len(l)):
                print i
                index = result.index(self.vars[i])
                lb = result[index].lb()
                ub = result[index].ub()
                if l[i] < lb or l[i] > ub:
                    l[i] = (lb+ub)/2
            if print_:
                print "Updated coefficients:"
                print l
            return l

        # neural network update
        layers = self.model.children()
        updated = []
        for k,l in enumerate(layers):

            if print_:
                print "Old weights of layer %d" %(k+1)
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
                print "Updated weights of layer %d" %(k+1)
                print updated[-1]

        return updated


    #small helper function
    def discount(self, rewards, gamma):
        discounted_rewards = [0]
        for r in rewards[::-1]:
            discounted_rewards = [r + gamma*discounted_rewards[0]] + discounted_rewards
        return discounted_rewards
