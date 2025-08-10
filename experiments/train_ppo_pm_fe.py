import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from features_extractor import GoalMinigridFeaturesExtractor_PM
from minigrid_envs import MetaSimpleEnvReachOneGoal, DictGoalsWrappers
from features_extractor import GoalActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from joblib import Parallel, delayed
import itertools


def train_single_seed(seed, arch):
    """Train a single model with the given seed."""
    policy_kwargs = dict(
        net_arch=[32],
        share_features_extractor=True,
        features_extractor_class=GoalMinigridFeaturesExtractor_PM,
        features_extractor_kwargs=dict(features_dim1=arch[0], features_dim2=arch[1]),
    )
    env = Monitor(DictGoalsWrappers(ImgObsWrapper(MetaSimpleEnvReachOneGoal(seed_goal_number=0))), f'MPFE_{arch[0]}/gcrl_seed_{seed}/')
    model = PPO(GoalActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=0, device='cpu')
    model.learn(1e7)
    model.save(f'MPFE_{arch[0]}/gcrl_seed_{seed}/model')
    return f"Training completed for seed {seed} with architecture {arch}"



def train_all_combinations():
    """Train all combinations of seeds and architectures in parallel."""
    # Define architectures and seeds
    architectures = [(128, 128), (256, 64), (512, 32), (1024, 16), (2048, 8), (4096, 4), (8192, 2)]
    seeds = range(10)
    
    # Create all combinations of seeds and architectures
    combinations = list(itertools.product(seeds, architectures))
    
    
    # Parallelize over all combinations
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_single_seed)(seed, arch) for seed, arch in combinations
    )
    
    return results


    # Train all combinations in parallel
all_results = train_all_combinations()
print("All training completed!")
print(f"Total combinations trained: {len(all_results)}")

