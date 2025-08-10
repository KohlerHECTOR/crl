from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch
from stable_baselines3 import SAC
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from joblib import Parallel, delayed
from math import ceil, floor


def calculate_balanced_dims(target_bottleneck, obs_dim, action_dim, target_params=None, reference_dims=(256, 64)):
    """
    Calculate the first layer size (dims[0]) to maintain the same number of parameters
    when changing the bottleneck size (dims[1]).
    
    Args:
        target_bottleneck: The desired bottleneck size (dims[1])
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        target_params: Target number of parameters (if None, uses reference_dims to calculate)
        reference_dims: Reference architecture to calculate target parameters
    
    Returns:
        tuple: (dims[0], target_bottleneck) where dims[0] is calculated to maintain parameter count
    """
    # Calculate target parameters from reference architecture
    if target_params is None:
        ref_dims0, ref_dims1 = reference_dims
        # Based on the actual parameter structure observed:
        # Feature extractor parameters (2 copies: actor + critic_target):
        #   - First layer: obs_dim * dims0 + dims0
        #   - Second layer: dims0 * dims1 + dims1
        # Actor network parameters:
        #   - latent_pi: dims1 * 128 + 128
        #   - mu: 128 * action_dim + action_dim
        #   - log_std: 128 * action_dim + action_dim
        # Critic network parameters (2 critics + 2 targets):
        #   - qf0/qf1: (dims1 + action_dim) * 128 + 128 + 128 * 1 + 1
        #   - 4 total critics (2 main + 2 target)
        target_params = (2 * (obs_dim * ref_dims0 + ref_dims0 + ref_dims0 * ref_dims1 + ref_dims1) +  # 2 feature extractors
                        ref_dims1 * 128 + 128 +  # actor latent_pi
                        128 * action_dim + action_dim +  # actor mu
                        128 * action_dim + action_dim +  # actor log_std
                        4 * ((ref_dims1 + action_dim) * 128 + 128 + 128 * 1 + 1))  # 4 critics
    
    # For the target bottleneck, we need to solve:
    # 2 * (obs_dim * dims0 + dims0 + dims0 * target_bottleneck + target_bottleneck) +  # 2 feature extractors
    # target_bottleneck * 128 + 128 +  # actor latent_pi
    # 128 * action_dim + action_dim +  # actor mu
    # 128 * action_dim + action_dim +  # actor log_std
    # 4 * ((target_bottleneck + action_dim) * 128 + 128 + 128 * 1 + 1) = target_params  # 4 critics
    
    # Rearranging: dims0 * (2 * obs_dim + 2 + 2 * target_bottleneck) = target_params - 2 * target_bottleneck - target_bottleneck * 128 - 128 - 2 * (128 * action_dim + action_dim) - 4 * (action_dim * 128 + 128 + 128 * 1 + 1)
    # dims0 = (target_params - 2 * target_bottleneck - target_bottleneck * 128 - 128 - 2 * (128 * action_dim + action_dim) - 4 * (action_dim * 128 + 128 + 128 * 1 + 1)) / (2 * obs_dim + 2 + 2 * target_bottleneck)
    
    # Calculate fixed parameters (actor mu/log_std + critic fixed parts)
    fixed_params = (128 * action_dim + action_dim +  # actor mu
                   128 * action_dim + action_dim +  # actor log_std
                   4 * (action_dim * 128 + 128 + 128 * 1 + 1))  # 4 critics fixed parts
    
    # Calculate variable parameters (feature extractors + actor latent_pi + critic variable parts)
    variable_params = target_params - fixed_params
    
    # Solve for dims0 in feature extractors and variable parts
    # 2 * (obs_dim * dims0 + dims0 + dims0 * target_bottleneck + target_bottleneck) + target_bottleneck * 128 + 128 + 4 * (target_bottleneck * 128) = variable_params
    # 2 * dims0 * (obs_dim + 1 + target_bottleneck) + 2 * target_bottleneck + target_bottleneck * 128 + 128 + 4 * target_bottleneck * 128 = variable_params
    # dims0 * (2 * obs_dim + 2 + 2 * target_bottleneck) = variable_params - 2 * target_bottleneck - target_bottleneck * 128 - 128 - 4 * target_bottleneck * 128
    # dims0 = (variable_params - 2 * target_bottleneck - target_bottleneck * 128 - 128 - 4 * target_bottleneck * 128) / (2 * obs_dim + 2 + 2 * target_bottleneck)
    
    numerator = variable_params - 2 * target_bottleneck - target_bottleneck * 128 - 128 - 4 * target_bottleneck * 128
    denominator = 2 * obs_dim + 2 + 2 * target_bottleneck
    
    dims0 = ceil(numerator / denominator)
    
    # Ensure dims0 is at least 1
    dims0 = max(1, dims0)
    
    return (dims0, target_bottleneck)


def get_balanced_architectures(obs_dim, action_dim, reference_dims=(256, 64)):
    """
    Generate a list of architectures with the same number of parameters
    but different bottleneck sizes.
    
    Args:
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        reference_dims: Reference architecture to calculate target parameters
    
    Returns:
        list: List of (dims[0], dims[1]) tuples with balanced parameters
    """
    bottleneck_sizes = [128, 64, 32, 16, 8, 4, 2]
    architectures = []
    
    for bottleneck in bottleneck_sizes:
        balanced_dims = calculate_balanced_dims(bottleneck, obs_dim, action_dim, reference_dims=reference_dims)
        architectures.append(balanced_dims)
    
    return architectures


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
    # param_count = len(model.policy.parameters_to_vector())
    # print(f"Architecture {dims}: {param_count} parameters")
    model.learn(1e7)
    model.save(f'{env_name}/arch_{dims[0]}/seed_{seed}/model')

# Calculate balanced architectures for each environment
def get_balanced_architectures_for_env(env_name, reference_dims=(256, 64)):
    """Get balanced architectures for a specific environment"""
    # Create a temporary environment to get observation and action space dimensions
    temp_env = gym.make(f'{env_name}-v5')
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    architectures = get_balanced_architectures(obs_dim, action_dim, reference_dims)
    
    # Calculate expected parameter counts for verification
    print(f"\nExpected parameter counts for {env_name} (obs_dim={obs_dim}, action_dim={action_dim}):")
    ref_dims0, ref_dims1 = reference_dims
    
    # Calculate target parameters based on actual structure
    target_params = (2 * (obs_dim * ref_dims0 + ref_dims0 + ref_dims0 * ref_dims1 + ref_dims1) +  # 2 feature extractors
                    ref_dims1 * 128 + 128 +  # actor latent_pi
                    128 * action_dim + action_dim +  # actor mu
                    128 * action_dim + action_dim +  # actor log_std
                    4 * ((ref_dims1 + action_dim) * 128 + 128 + 128 * 1 + 1))  # 4 critics
    
    print(f"Target parameters: {target_params}")
    
    for dims in architectures:
        expected_params = (2 * (obs_dim * dims[0] + dims[0] + dims[0] * dims[1] + dims[1]) +  # 2 feature extractors
                          dims[1] * 128 + 128 +  # actor latent_pi
                          128 * action_dim + action_dim +  # actor mu
                          128 * action_dim + action_dim +  # actor log_std
                          4 * ((dims[1] + action_dim) * 128 + 128 + 128 * 1 + 1))  # 4 critics
        print(f"Architecture {dims}: expected {expected_params} parameters")
    
    return architectures

# Test the parameter balancing with a simple example
def test_parameter_balancing():
    """Test the parameter balancing with a simple example"""
    print("Testing parameter balancing...")
    
    # Test with Hopper environment
    env = gym.make('Hopper-v5')
    obs_dim = env.observation_space.shape[0]  # 11
    action_dim = env.action_space.shape[0]    # 3
    env.close()
    
    print(f"Hopper: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Get balanced architectures
    balanced_architectures = get_balanced_architectures(obs_dim, action_dim, reference_dims=(256, 64))
    print(f"Balanced architectures: {balanced_architectures}")
    
    # Test the balanced architectures
    for dims in balanced_architectures:
        policy_kwargs = dict(features_extractor_class=MLPFeats,
                            share_features_extractor=True,
                            features_extractor_kwargs=dict(features_dims=dims),
                            net_arch=[128])
        
        model = SAC('MlpPolicy', env, verbose=0, device='cpu', policy_kwargs=policy_kwargs)
        param_count = len(model.policy.parameters_to_vector())
        print(f"Architecture {dims}: {param_count} parameters")
        
        # Print detailed parameter breakdown for the first one
        if dims == balanced_architectures[0]:
            print("  Parameter breakdown:")
            for name, param in model.policy.named_parameters():
                print(f"    {name}: {param.shape} ({param.numel()} parameters)")
            print()

# Run the test
test_parameter_balancing()

# Generate balanced architectures for each environment
env_names = ['Ant', 'Hopper', 'HalfCheetah', 'Walker2d']
balanced_architectures = {}

for env_name in env_names:
    balanced_architectures[env_name] = get_balanced_architectures_for_env(env_name)

# Print the balanced architectures for verification
print("Balanced architectures for each environment:")
for env_name, archs in balanced_architectures.items():
    print(f"{env_name}: {archs}")

# Parallelize the loops with balanced architectures
Parallel(n_jobs=-1)(
    delayed(run_experiment)(seed, dims, env_name)
    for seed in range(10)
    for env_name in env_names
    for dims in balanced_architectures[env_name]
)