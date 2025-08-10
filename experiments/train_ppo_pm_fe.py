import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from features_extractor import GoalMinigridFeaturesExtractor_PM
from minigrid_envs import MetaSimpleEnvReachOneGoal, DictGoalsWrappers
from features_extractor import GoalActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from joblib import Parallel, delayed
import itertools
from math import ceil
import torch


def calculate_balanced_dims(target_bottleneck, n_flatten, target_params=None, reference_dims=(512, 512)):
    """
    Calculate the first layer size (dims[0]) to maintain the same number of parameters
    when changing the bottleneck size (dims[1]).
    
    Args:
        target_bottleneck: The desired bottleneck size (dims[1])
        n_flatten: Number of flattened features from CNN
        target_params: Target number of parameters (if None, uses reference_dims to calculate)
        reference_dims: Reference architecture to calculate target parameters
    
    Returns:
        tuple: (dims[0], target_bottleneck) where dims[0] is calculated to maintain parameter count
    """
    # Calculate target parameters from reference architecture
    if target_params is None:
        ref_dims0, ref_dims1 = reference_dims
        # Based on the actual parameter structure observed:
        # CNN parameters (fixed):
        #   - Conv2d(16, 3, 2, 2): 16 * 3 * 2 * 2 + 16 = 208
        #   - Conv2d(32, 16, 2, 2): 32 * 16 * 2 * 2 + 32 = 2080
        #   - Conv2d(64, 32, 2, 2): 64 * 32 * 2 * 2 + 64 = 8256
        # Feature extractor linear layers:
        #   - First linear layer: n_flatten * dims0 + dims0
        #   - Second linear layer: dims0 * dims1 + dims1
        # Policy/Value networks (16 tasks each):
        #   - 16 policy nets: 16 * (dims1 * 32 + 32)
        #   - 16 value nets: 16 * (dims1 * 32 + 32)
        # Final action/value nets:
        #   - action_net: 7 * 32 + 7
        #   - value_net: 1 * 32 + 1
        
        cnn_params = 208 + 2080 + 8256  # Fixed CNN parameters
        target_params = (cnn_params +
                        n_flatten * ref_dims0 + ref_dims0 +  # first linear layer
                        ref_dims0 * ref_dims1 + ref_dims1 +  # second linear layer
                        16 * (ref_dims1 * 32 + 32) +  # 16 policy nets
                        16 * (ref_dims1 * 32 + 32) +  # 16 value nets
                        7 * 32 + 7 +  # action_net
                        1 * 32 + 1)   # value_net
    
    # For the target bottleneck, we need to solve:
    # cnn_params + n_flatten * dims0 + dims0 + dims0 * target_bottleneck + target_bottleneck + 
    # 16 * (target_bottleneck * 32 + 32) + 16 * (target_bottleneck * 32 + 32) + 
    # 7 * 32 + 7 + 1 * 32 + 1 = target_params
    
    # Calculate fixed parameters (CNN + final nets)
    cnn_params = 208 + 2080 + 8256
    fixed_params = cnn_params + 7 * 32 + 7 + 1 * 32 + 1
    
    # Calculate variable parameters (feature extractor + policy/value nets)
    variable_params = target_params - fixed_params
    
    # Solve for dims0 in feature extractor and policy/value nets
    # n_flatten * dims0 + dims0 + dims0 * target_bottleneck + target_bottleneck + 
    # 32 * (target_bottleneck * 32 + 32) = variable_params
    # dims0 * (n_flatten + 1 + target_bottleneck) + target_bottleneck + 32 * target_bottleneck * 32 + 32 * 32 = variable_params
    # dims0 * (n_flatten + 1 + target_bottleneck) = variable_params - target_bottleneck - 32 * target_bottleneck * 32 - 32 * 32
    # dims0 = (variable_params - target_bottleneck - 32 * target_bottleneck * 32 - 32 * 32) / (n_flatten + 1 + target_bottleneck)
    
    numerator = variable_params - target_bottleneck - 32 * target_bottleneck * 32 - 32 * 32
    denominator = n_flatten + 1 + target_bottleneck
    
    dims0 = ceil(numerator / denominator)
    
    # Ensure dims0 is at least 1
    dims0 = max(1, dims0)
    
    return (dims0, target_bottleneck)


def get_balanced_architectures(n_flatten, reference_dims=(512, 512)):
    """
    Generate a list of architectures with the same number of parameters
    but different bottleneck sizes.
    
    Args:
        n_flatten: Number of flattened features from CNN
        reference_dims: Reference architecture to calculate target parameters
    
    Returns:
        list: List of (dims[0], dims[1]) tuples with balanced parameters
    """
    bottleneck_sizes = [128, 64, 32, 16, 8, 4, 2]
    architectures = []
    
    for bottleneck in bottleneck_sizes:
        balanced_dims = calculate_balanced_dims(bottleneck, n_flatten, reference_dims=reference_dims)
        architectures.append(balanced_dims)
    
    return architectures


def get_balanced_architectures_for_env(reference_dims=(512, 512)):
    """Get balanced architectures for the minigrid environment"""
    # Create a temporary environment to calculate n_flatten
    temp_env = Monitor(DictGoalsWrappers(ImgObsWrapper(MetaSimpleEnvReachOneGoal(seed_goal_number=0))))
    
    # Get observation space info
    obs_space = temp_env.observation_space
    print(f"Observation space: {obs_space}")
    
    # Extract the actual observation space from the Dict
    if hasattr(obs_space, 'spaces') and 'observation' in obs_space.spaces:
        actual_obs_space = obs_space.spaces['observation']
        print(f"Actual observation space: {actual_obs_space}")
        print(f"Actual observation space shape: {actual_obs_space.shape}")
    else:
        # Fallback: assume it's a flattened observation
        n_flatten = 64  # Default value for flattened features
        print(f"Using default n_flatten: {n_flatten}")
        temp_env.close()
        return get_balanced_architectures(n_flatten, reference_dims)
    
    # Calculate n_flatten manually based on the CNN architecture
    # CNN: Conv2d(16, (2,2)) -> Conv2d(32, (2,2)) -> Conv2d(64, (2,2)) -> Flatten
    # Input: actual_obs_space.shape[0] x actual_obs_space.shape[1] x actual_obs_space.shape[2] (height x width x channels)
    
    # For the CNN architecture in GoalMinigridFeaturesExtractor_PM:
    # Input: (height, width, channels) - e.g., (7, 7, 3)
    # Conv2d(16, (2,2)): height -> height-1, width -> width-1
    # Conv2d(32, (2,2)): height -> height-1, width -> width-1  
    # Conv2d(64, (2,2)): height -> height-1, width -> width-1
    # So final spatial size: (height-3) x (width-3)
    
    # Get the spatial dimensions from the actual observation space
    if hasattr(actual_obs_space, 'shape'):
        if len(actual_obs_space.shape) == 3:  # (height, width, channels)
            height, width, channels = actual_obs_space.shape
        elif len(actual_obs_space.shape) == 2:  # (height, width) - assume 1 channel
            height, width = actual_obs_space.shape
            channels = 1
        else:
            # Fallback: assume it's a flattened observation
            n_flatten = 1024  # Default value for flattened features
            print(f"Using default n_flatten: {n_flatten}")
            temp_env.close()
            return get_balanced_architectures(n_flatten, reference_dims)
    else:
        # Fallback: assume it's a flattened observation
        n_flatten = 1024  # Default value for flattened features
        print(f"Using default n_flatten: {n_flatten}")
        temp_env.close()
        return get_balanced_architectures(n_flatten, reference_dims)
    
    # Calculate final spatial dimensions after CNN
    # Input is (7, 7, 3) which means (height=7, width=7, channels=3)
    # After 3 Conv2d layers with (2,2) kernels: height -> height-3, width -> width-3
    final_height = max(1, height - 3)
    final_width = max(1, width - 3)
    n_flatten = 64 * final_height * final_width  # 64 channels from last conv layer
    
    print(f"Calculated n_flatten: {n_flatten} (from {height}x{width}x{channels} -> {final_height}x{final_width}x64)")
    
    temp_env.close()
    
    architectures = get_balanced_architectures(n_flatten, reference_dims)
    
    # Calculate expected parameter counts for verification
    print(f"\nExpected parameter counts for minigrid (n_flatten={n_flatten}):")
    ref_dims0, ref_dims1 = reference_dims
    target_params = (n_flatten * ref_dims0 + ref_dims0 +  # first linear layer
                    ref_dims0 * ref_dims1 + ref_dims1 +  # second linear layer
                    16 * (ref_dims1 * 32 + 32) +  # 16 policy nets
                    16 * (ref_dims1 * 32 + 32) +  # 16 value nets
                    16 * (32 * 7 + 7) +  # 16 policy outputs
                    16 * (32 * 1 + 1))   # 16 value outputs
    
    print(f"Target parameters: {target_params}")
    
    for dims in architectures:
        expected_params = (n_flatten * dims[0] + dims[0] +  # first linear layer
                          dims[0] * dims[1] + dims[1] +  # second linear layer
                          16 * (dims[1] * 32 + 32) +  # 16 policy nets
                          16 * (dims[1] * 32 + 32) +  # 16 value nets
                          16 * (32 * 7 + 7) +  # 16 policy outputs
                          16 * (32 * 1 + 1))   # 16 value outputs
        print(f"Architecture {dims}: expected {expected_params} parameters")
    
    return architectures


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
    # Get balanced architectures
    balanced_architectures = get_balanced_architectures_for_env(reference_dims=(512, 512))
    print(f"Balanced architectures: {balanced_architectures}")
    
    # Define seeds
    seeds = range(10)
    
    # Create all combinations of seeds and architectures
    combinations = list(itertools.product(seeds, balanced_architectures))
    
    # Parallelize over all combinations
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_single_seed)(seed, arch) for seed, arch in combinations
    )
    
    return results

# Train all combinations in parallel
all_results = train_all_combinations()
print("All training completed!")
print(f"Total combinations trained: {len(all_results)}")

