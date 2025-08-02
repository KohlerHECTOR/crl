import torch 
from minigrid_envs import MetaSimpleEnvReachOneGoal, DictGoalsWrappers
from minigrid.wrappers import ImgObsWrapper
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from time import sleep


# per goal correlation of features and overall correlations

model = PPO.load(f'policies/bigger_head_gcrl_seed_29')

env = DictGoalsWrappers(ImgObsWrapper(MetaSimpleEnvReachOneGoal(seed_goal_number=0, max_steps=50)))

success = 0
all_feats = torch.zeros((10_000, 128))
done = False
s, _= env.reset()
ep_norms_1 = []
ep_norm_1 = []
ep_norms_2 = []
ep_norm_2 = []
for t in range(10_000):
    o = model.policy.obs_to_tensor(s)[0]
    with torch.no_grad():
        all_feats[t]= model.policy.extract_features(o)
    ep_norm_1.append(all_feats[t].max())
    # print(torch.sort(all_feats[t])[0][-10:])
    
    a = model.predict(s, deterministic=False)[0] # stochastic because POMDP
    s, r, term, trunc, _ = env.step(a)
    done = term or trunc
    if done:
        s, _ = env.reset()
        ep_norms_1.append(ep_norm_1)
        ep_norm_1 = []

for n in ep_norms_1:
    plt.plot(n)
plt.show()
plt.clf()

# for n in ep_norms_2:
#     plt.plot(n)
# plt.show()
# plt.clf()

# argmaxes_ = all_feats.argmax(dim=0)
# print(len(argmaxes_))
# for idx in argmaxes_:
#     plt.bar(range(128), height=all_feats[idx], width=2)
#     plt.show()
#     plt.clf()