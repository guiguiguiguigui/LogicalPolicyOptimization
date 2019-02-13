import numpy as np
import gym

env = gym.make('CartPole-v1')

ob = env.reset()
ob_dtype = ob.dtype
ob_len = len(ob)

ob_as_byte = ob.tobytes()

# Save data to test whether restoring state works
test_stats = {}

# Sampling data from the environment
for i in range(100000):

    assert ob_as_byte not in test_stats

    act = env.action_space.sample()

    ob, rew, done, _ = env.step(act)

    test_stats[ob_as_byte] = [act, ob, rew, done]

    if done:
        ob = env.reset()

    ob_as_byte = ob.tobytes()
# TEST
# We test that given the same observation (state), taking the same action
# will bring us to the same next observation, reward and done info
for ob_as_byte in test_stats.keys():

    act, exp_ob, exp_rew, exp_done = test_stats[ob_as_byte]  # exp stands for expected

    # We reset the environments so that done is not carried over from the last for loop
    env.reset()

    # Restore the env state
    # In CartPole, the observation and state are the same thing
    # https://github.com/openai/gym/blob/49cd48020f6760630a7317cb3529a22de6f12f2e/gym/envs/classic_control/cartpole.py#L130
    env.env.state = np.frombuffer(ob_as_byte, dtype=ob_dtype, count=ob_len)

    ob, rew, done, _ = env.step(act)

    assert (ob == exp_ob).all(), (ob, exp_ob)
    assert rew == exp_rew, (rew, exp_rew)
    assert done == exp_done, (done, exp_done)

print('Finished Testing')