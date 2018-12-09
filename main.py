import argparse, math, os
import numpy as np
import gym
from gym import wrappers

import torch
import torch.nn.utils as utils

from normalized_actions import NormalizedActions

from policies.policy_1layer import SingleLayerPolicy
from policies.policy_2layer import TwoLayerPolicy
from value_functions import nnValueFunction

parser = argparse.ArgumentParser()
# model & env selection
parser.add_argument('--env_name', type=str, default='CartPole-v0')
parser.add_argument('--gradient', type=bool, default=False, help='use gradient or not') 
# hyperparameters 
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, metavar='N',help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N', help='max episode length (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=15, metavar='N', help='number of hidden neurons (default: 100)')
parser.add_argument('--layers', type=int, default=2, metavar='N', help='number of layers inf the policy NN')
# LPO parameters
parser.add_argument('--constraint_size',type=int, default=10, metavar='N', help='number of constraint to be solved each time')
parser.add_argument('--iter', type=int, default=10, metavar='N', help='number of iterations of solving x constraits')
# Save & render 
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--ckpt_freq', type=int, default=2, help='model saving frequency')
parser.add_argument('--display', type=bool, default=False, help='display or not')
args = parser.parse_args()


'''
Initiate enviornment
'''
env_name = args.env_name
env = gym.make(env_name)
if type(env.action_space) != gym.spaces.discrete.Discrete:
    from LPO_continuous import LPO
    env = NormalizedActions(gym.make(env_name))
else:
    from LPO_discrete import LPO

if args.display:
    env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(env_name), force=True)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

ckpt = 'ckpt_' + env_name
if not os.path.exists(ckpt):    
    os.mkdir(ckpt)

'''
Create policy, value function, and agent
'''
if args.layers == 1:
    policy = SingleLayerPolicy(args.hidden_size, env.observation_space.shape[0], env.action_space)
elif args.layers== 2:
    policy = TwoLayerPolicy(args.hidden_size, env.observation_space.shape[0], env.action_space)

agent = LPO(args.hidden_size, env.observation_space.shape[0], env.action_space, args.constraint_size, policy)
vf = nnValueFunction(ob_dim=env.observation_space.shape[0])


'''
Here we gooooooo
'''
def main():
    
    for i_iter in range(args.iter):
        print("\niteration %d \n" %i_iter)
        count_rollouts = 0
        
        while True: #keep doing rollouts untill reach the # of constraints
            
            start_state = env.reset()
            state = torch.Tensor([start_state])
            
            entropies, log_probs = [], [] # currently not looked at, can be useful tho
            obs, acs, rewards = [start_state], [], []

            for t in range(args.num_steps):
                action, log_prob, entropy = agent.select_action(state)
                acs.append(action)
                action = action.cpu()

                next_state, reward, rollout_done, _ = env.step(action.numpy()[0])
                obs.append(next_state)
                rewards.append(reward)
                entropies.append(entropy)
                log_probs.append(log_prob)
                state = torch.Tensor([next_state])

                if rollout_done:
                    break

            # Estimate advantage function using baseline vf (these are lists!).
            discounted_rewards = torch.Tensor(agent.discount(rewards, args.gamma))
            vpreds = vf(torch.Tensor(obs)) #list of value function's predictions of obs
            advs = discounted_rewards - vpreds

            #re-fit value function baseline
            vf.fit(torch.Tensor(obs), discounted_rewards)

            # policy update! (if we have enough constraints)
            std_adv_n = (advs - advs.mean()) / (advs.std() + 1e-8)
            iter_done = agent.update_parameters(obs, acs, std_adv_n.detach().numpy())
            if iter_done:
                break
            
            print("Iter: {}, Rollout: {}, reward: {}".format(i_iter, count_rollouts, np.sum(rewards)))
            count_rollouts += 1
        
        if i_iter%args.ckpt_freq == 0:
               torch.save(agent.model.state_dict(), os.path.join(ckpt, 'lpo-'+str(i_iter)+'.pkl'))

    env.close()

if __name__ == '__main__':
    main()
