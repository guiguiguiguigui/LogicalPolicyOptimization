# LogicalPolicyOptimization

PyTorch implementation in development.     
This repo currently only supports the **CartPole-v0** enviornment. Other continuous and discrete envs in OpenAI gym is comming soon. 

## Requirement
- python 2.7
- PyTorch
- OpenAI gym
- Mujoco (optional)


## Run
Use the default hyperparameters.

```
python main.py --env_name CartPole-v0 --num_rollout 50 --hidden_size 10 --constraint_size 1
```

## Reference
- [pytorch example](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)
