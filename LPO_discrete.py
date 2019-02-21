import math
import numpy as np
from copy import copy
import time

import dreal

class LPO:
    def __init__(self, hidden_size, num_inputs, action_space, constraint_size, policy):
        self.action_space = action_space
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.constraint_size = constraint_size
        self.model = policy
        self.true_= dreal.Expression(1) > dreal.Expression(0) #a hack
        self.false_= dreal.Expression(1) < dreal.Expression(0) #a hack
        self.vars, self.limits = self.model.generate_variable_limits(hidden_size, num_inputs, action_space, [-50,50])

        self.empty_constraints = False
        self.constraints = []
        self.training_samples = {"states": np.array([]).reshape(0,4),
                                  "advantage":np.array([]), 
                                  "actions":np.array([]).astype(int)}

    def select_action(self, state):
        probs = self.model(state)
        # TODO: this only works in th cartpole case
        action = np.where(np.random.multinomial(1,probs)==1)[0][0]
        action = int(action)
        # TODO: implement
        log_prob, entropy = 0,0
        #log_prob = prob.log()
        #entropy = - (probs*probs.log()).sum()
        return action, log_prob, entropy

    def update_parameters(self, states, actions, advantage):

        rounded_states = np.around(states[1:], 2) #to be experimented with

        for i, i_state in enumerate(rounded_states[:-1]):
            for j, j_state in enumerate(rounded_states[i+1:]):
                if np.all(np.equal(i_state, j_state)) and (actions[i] != actions[j]) and (advantage[i] != advantage[j]):
                    #need to explicitly formulate these things 
                    print "constraint %d! state: " % (len(self.constraints)+1)
                    print i_state
                    #prob = self.model.forward_symbolic(self.vars, i_state)
                    #c = ((prob[actions[i]] - prob[actions[j]]) * (advantage[i] - advantage[j]) > 0)

                    prob_1 = self.model.forward_symbolic_y(self.vars, i_state)
                    if actions[i] == 1:
                        c = ((advantage[i] - advantage[j])*prob_1 > 0)
                    elif actions[j] == 1:
                        c = ((advantage[j] - advantage[i])*prob_1 > 0)
                    else:
                        print "something's wrong here with constraints"
                        c = self.true_
                    if c == self.false_:
                        print "false here."
                        print actions[i], advantage[i]
                        print actions[j], advantage[j]

                    print c
                    self.constraints.append(c)

            for j, j_state in enumerate(self.training_samples["states"]):
                if np.all(np.equal(i_state, j_state)) and (actions[i] != self.training_samples["actions"][j]) and (advantage[i] != self.training_samples["advantage"][j]):
                    #need to explicitly formulate these things 
                    print "constraint %d! state: " % (len(self.constraints)+1)
                    print i_state
                    #prob = self.model.forward_symbolic(self.vars, i_state)
                    #c = ((prob[actions[i]] - prob[self.training_samples["actions"][j]]) * \
                    #(advantage[i] - self.training_samples["advantage"][j]) > 0)
                    
                    prob_1 = self.model.forward_symbolic_y(self.vars, i_state)
                    if actions[i] == 1:
                        c = ((advantage[i] - self.training_samples["advantage"][j])*prob_1 > 0)
                    elif self.training_samples["actions"][j] == 1:
                        c = ((self.training_samples["advantage"][j] - advantage[i])*prob_1 > 0)
                    else:
                        print "something's wrong here with constraints"
                        c = self.true_
                    if c == self.false_: 
                        print "false here."
                        print actions[i], advantage[i]
                        print self.training_samples["actions"][j], self.training_samples["advantage"][j]


                    print c
                    self.constraints.append(c)

        if len(self.constraints) < self.constraint_size:
            #If there are not enough traing samples, we postpone training to next iteration
            self.training_samples["states"] = np.concatenate([self.training_samples["states"],rounded_states])
            self.training_samples["advantage"] = np.concatenate([self.training_samples["advantage"], advantage])
            # hack. if the sequence terminates, there will be 1 more state than action
            # we pad the action sequence such that their indices align 
            actions_padding = [-1]*(len(rounded_states)-len(actions))

            self.training_samples["actions"] = np.concatenate([self.training_samples["actions"],actions,actions_padding]).astype(int)
            assert(len(self.training_samples["states"]) == len(self.training_samples["advantage"]))
            assert(len(self.training_samples["states"]) == len(self.training_samples["actions"]))

            return False

        else:
            print("# of Constraints: %d "%len(self.constraints))
            all_constraints = self.constraints +self.limits 
            #print constraints
            print("optimizer starts")
            #print(constraints)
            #print constraints

            f_sat = dreal.logical_and(*all_constraints)
            timer_start = time.time()
            result = dreal.CheckSatisfiability(f_sat, 0.0001)
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
            if self.empty_constraints:
                self.constraints = [] 
                self.training_samples = {"states": np.array([]).reshape(0,4), 
                                        "advantage":np.array([]), 
                                        "actions":np.array([]).astype(int)}
            else:
                self.constraint_size += self.constraint_size
            return True
    
    def get_weights_from_result(self, result, print_=True):
        if print_:
            print "Check_Satisfiability result:"
            print result

        l = self.model.coef
        if print_:
            print "Old coefficients:"
            print l #old weights
        updated = []
        for i in range(len(l)):
            index = result.index(self.vars[i])
            lb = result[index].lb()
            ub = result[index].ub()
            if l[i] < lb or l[i] > ub:
                l[i] = (lb+ub)/2
        if print_:
            print "Updated coefficients:"
            print l
        return l

        return updated


    #small helper function
    def discount(self, rewards, gamma):
        discounted_rewards = [0]
        for r in rewards[::-1]:
            discounted_rewards = [r + gamma*discounted_rewards[0]] + discounted_rewards
        return discounted_rewards[:-1]