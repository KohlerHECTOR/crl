import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from features_extractor import GoalMinigridFeaturesExtractor
from minigrid_envs import MetaSimpleEnvReachOneGoal, DictGoalsWrappers
from features_extractor import GoalActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from joblib import Parallel, delayed
import os

policy_kwargs = dict(
    net_arch=[128],
    share_features_extractor=True,
    features_extractor_class=GoalMinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)


def train_single_seed(seed):
    """Train a single model with the given seed."""
    env = Monitor(DictGoalsWrappers(ImgObsWrapper(MetaSimpleEnvReachOneGoal(seed_goal_number=0))), f'gcrl_seed_{seed}/')
    model = PPO(GoalActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=0, device='cpu')
    model.learn(1e7)
    model.save(f'gcrl_seed_{seed}/model.pkl')
    return f"Training completed for seed {seed}"


# Parallelize training across 100 seeds
# Using n_jobs=-1 to use all available CPU cores
# You can adjust n_jobs to a specific number if needed
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_single_seed)(s) for s in range(70)
)

