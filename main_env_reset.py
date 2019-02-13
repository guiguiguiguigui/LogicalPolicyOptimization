import argparse, math, os
import numpy as np
import gym
from gym import wrappers

import numpy as np

import torch
import torch.nn.utils as utils

from normalized_actions import NormalizedActions

from policies.policy_1layer import SingleLayerPolicy
from policies.policy_2layer import TwoLayerPolicy

parser = argparse.ArgumentParser()
# model & env selection
parser.add_argument('--env_name', type=str, default='CartPole-v0')
parser.add_argument('--gradient', type=bool, default=False, help='use gradient or not')
# hyperparameters 
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, metavar='N',help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N', help='max episode length (default: 1000)')
parser.add_argument('--num_rollouts', type=int, default=2000, metavar='N', help='number of rollouts (default: 2000)')
parser.add_argument('--hidden_size', type=int, default=15, metavar='N', help='number of hidden neurons (default: 100)')
parser.add_argument('--constraint_size',type=int, default=10, metavar='N', help='number of constraint to be solved each time')
parser.add_argument('--layers', type=int, default=2, metavar='N', help='number of layers inf the policy NN')
# Save & render 
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--ckpt_freq', type=int, default=200, help='model saving frequency')
parser.add_argument('--display', type=bool, default=False, help='display or not')
args = parser.parse_args()

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

if args.layers == 1:
    policy = SingleLayerPolicy(args.hidden_size, env.observation_space.shape[0], env.action_space)
elif args.layers== 2:
    policy = TwoLayerPolicy(args.hidden_size, env.observation_space.shape[0], env.action_space)
agent = LPO(args.hidden_size, env.observation_space.shape[0], env.action_space, args.constraint_size, policy)

dir = 'ckpt_' + env_name
if not os.path.exists(dir):    
    os.mkdir(dir)

# change this. 
# sample 1 trajectory of n steps (n = numsteps: arg)
# for each of the n steps, sample k trajectories (k: arg)
# create constraints for these n*k constraints
for i_episode in range(args.num_rollouts):
    start_state = env.reset()
    state = torch.Tensor([start_state])
    entropies = []
    log_probs = []
    
    states = [start_state]
    states_bytes = [start_state.tobytes()]
    actions = []
    rewards = []

    for t in range(args.num_steps):
        action, log_prob, entropy = agent.select_action(state)
        actions.append(action)
        action = action.cpu()

        next_state, reward, done, _ = env.step(action.numpy()[0])
        states_bytes.append(next_state.tobytes)
        states.append(next_state)
        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            break
    #replay
    for st in states_bytes:
        for i in range()

    if args.gradient:
        agent.update_parameters_gradient(states, actions, rewards, log_probs, entropies, args.gamma)
    else:
        agent.update_parameters(states, actions, rewards, log_probs, entropies, args.gamma)

    if i_episode%args.ckpt_freq == 0:
       torch.save(agent.model.state_dict(), os.path.join(dir, 'lpo-'+str(i_episode)+'.pkl'))

    print("Rollout: {}, reward: {}".format(i_episode, np.sum(rewards)))
    
    if i_episode == 999:
        print(agent.model.linear1.weight)
        print(agent.model.linear2.weight)

env.close()


def sample_trajectories(env, start_state, state_byte, k):
    #trajectories = []
    actions = []
    discounted_rewards = []
    for rollout in range(k):
        #reset the state
        env.env.state = np.frombuffer(state_byte, dtype=state.dtype, count=len(start_state))
        actions = []
        rewards = []

        action = 0
        reward = 0

        state = torch.Tensor([start_state])

        for t in range(steps):
            action, log_prob, entropy = agent.select_action(state)
            if t==0:
                action = action
            action = action.cpu()

            next_state, reward, done, _ = env.step(action.numpy()[0])
            states_bytes.append(next_state.tobytes)
            states.append(next_state)
            rewards.append(reward)

            state = torch.Tensor([next_state])
            if done:
                break

        trajectories.append(trajectory)
        trajectory_actions.append(actions)
        trajectory_rewards.append(rewards)

        action_squashed = [a[0] for a in trajectory_actions]

        def discounted_rewards(gamma, r):
            aggregate = 0
            for r in r[::-1]:
                aggregate = r+gamma*aggregate
            return aggregate

        reward_squashed = [discounted_rewards(gamma,r) for r in trajectory_rewards]
        return action_squashed, reward_squashed


