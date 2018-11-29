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

a simple test run:
```
python main.py --num_rollout 50 --hidden_size 10 --constraint_size 1
```

running it for real:
```
python main.py --num_rollout 1000 --hidden_size 15 --constraint_size 10
```
## Reference
- [pytorch example](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)
