from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch
from stable_baselines3 import SAC
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from joblib import Parallel, delayed

class MLPFeats(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dims=(256, 64)):
        super().__init__(observation_space, features_dim=features_dims[1])
        self.model = nn.Sequential(nn.Linear(observation_space.shape[0], features_dims[0]), nn.ReLU(), nn.Linear(features_dims[0], features_dims[1]), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)


def run_experiment(seed, dims, env_name):
    env = Monitor(gym.make(f'{env_name}-v5'), f'{env_name}/arch_{dims[0]}/seed_{seed}/')

    policy_kwargs = dict(features_extractor_class=MLPFeats,
                        share_features_extractor=True,
                        features_extractor_kwargs=dict(features_dims=dims),
                        net_arch=[128])

    model = SAC('MlpPolicy', env, verbose=0, device='cpu', policy_kwargs=policy_kwargs)
    model.learn(1e7)
    model.save(f'{env_name}/arch_{dims[0]}/seed_{seed}/model')

# Parallelize the loops
Parallel(n_jobs=-1)(
    delayed(run_experiment)(seed, dims, env_name)
    for seed in range(10)
    for dims in [(128, 128), (256, 64), (512, 32), (1024, 16), (2048, 8), (4096, 4), (8192, 2)]
    for env_name in ['Ant', 'Hopper', 'HalfCheetah', 'Walker2d']
)