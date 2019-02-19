# LogicalPolicyOptimization - Linear

PyTorch implementation in development.     
This repo currently only supports the **CartPole-v0** enviornment. Other continuous and discrete envs in OpenAI gym is comming soon. 

## Requirement
- python 2.7
- PyTorch
- OpenAI gym
- Mujoco (optional)


## Run
Use the default hyperparameters.

a simple test run: (solving take ~2s)
```
python main.py --hidden_size 5 --gamma 0.99 --constraint_size 3 --iter 2
```

running it for real: (solving take ~40s)
```
python main.py --hidden_size 7 --gamma 0.99 --constraint_size 30 --iter 20
```

linear policy
```
python main.py --layers 0  --gamma 0.99 --constraint_size 30 --iter 20
```
## Reference
- [pytorch example](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)
