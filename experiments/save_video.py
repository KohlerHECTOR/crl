import torch 
from minigrid_envs import MetaSimpleEnvReachOneGoal, DictGoalsWrappers
from minigrid.wrappers import ImgObsWrapper
import numpy as np
from stable_baselines3 import PPO

model = PPO.load(f'policies/bigger_head_gcrl_seed_29')

env = DictGoalsWrappers(ImgObsWrapper(MetaSimpleEnvReachOneGoal(seed_goal_number=0, render_mode='human', max_steps=50)))

success = 0
for _ in range(1000):
    done = False
    s, _= env.reset()
    env.render()
    while not done:
        a = model.predict(s, deterministic=False)[0] # stochastic because POMDP
        s, r, term, trunc, _ = env.step(a)
        env.render()
        if r>0:
            success += 1
        done = term or trunc