from minatar import Environment
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch
from stable_baselines3 import DQN
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from joblib import Parallel, delayed
from gymnasium import spaces
import numpy as np
import seaborn as sns
from math import ceil


def calculate_balanced_dims(target_bottleneck, n_flatten, target_params=None, reference_dims=(128, 128)):
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
        # Parameters in linear layers: n_flatten * dims0 + dims0 + dims0 * dims1 + dims1 + dims1 * 128 + 128
        target_params = (n_flatten * ref_dims0 + ref_dims0 + 
                        ref_dims0 * ref_dims1 + ref_dims1 + 
                        ref_dims1 * 128 + 128)
    
    # For the target bottleneck, we need to solve:
    # n_flatten * dims0 + dims0 + dims0 * target_bottleneck + target_bottleneck + target_bottleneck * 128 + 128 = target_params
    
    # Rearranging: dims0 * (n_flatten + 1 + target_bottleneck) = target_params - target_bottleneck - target_bottleneck * 128 - 128
    # dims0 = (target_params - target_bottleneck - target_bottleneck * 128 - 128) / (n_flatten + 1 + target_bottleneck)
    
    numerator = target_params - target_bottleneck - target_bottleneck * 128 - 128
    denominator = n_flatten + 1 + target_bottleneck
    
    dims0 = ceil(numerator / denominator)
    
    # Ensure dims0 is at least 1
    dims0 = max(1, dims0)
    
    return (dims0, target_bottleneck)


def get_balanced_architectures(n_flatten, reference_dims=(128, 128)):
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


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "array", "rgb_array"]}

    def __init__(self, game, render_mode=None, display_time=50,
                use_minimal_action_set=False, **kwargs):
        self.render_mode = render_mode
        self.display_time = display_time

        self.game = Environment(env_name=game, **kwargs)

        if use_minimal_action_set:
            self.action_set = self.game.minimal_action_set()
        else:
            self.action_set = list(range(self.game.num_actions()))

        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(
            0, 1, shape=self.game.state_shape(), dtype=np.uint8
        )

    def step(self, action):
        action = self.action_set[action]
        reward, done = self.game.act(action)
        if self.render_mode == "human":
            self.render()
        return self.game.state(), reward, done, False, {}

    def seed(self, seed=None):
        self.game.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.game.reset()
        if self.render_mode == "human":
            self.render()
        return self.game.state(), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "array":
            return self.game.state()
        elif self.render_mode == "human":
            self.game.display_state(self.display_time)
        elif self.render_mode == "rgb_array": # use the same color palette of Environment.display_state
            state = self.game.state()
            n_channels = state.shape[-1]
            cmap = sns.color_palette("cubehelix", n_channels)
            cmap.insert(0, (0,0,0))
            numerical_state = np.amax(
                state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2)
            rgb_array = np.stack(cmap)[numerical_state]
            return rgb_array

    def close(self):
        if self.game.visualized:
            self.game.close_display()
        return 0

class MinatarFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, dims=(64, 64)) -> None:
        super().__init__(observation_space, dims[1])
        n_input_channels = observation_space.shape[-1]
        self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten())
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            # Permute dimensions from [batch, height, width, channels] to [batch, channels, height, width]
            sample_obs = sample_obs.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(in_features=n_flatten, out_features=dims[0]), nn.ReLU(), nn.Linear(dims[0], dims[1]), nn.ReLU())
        # Output layer:

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Permute dimensions from [batch, height, width, channels] to [batch, channels, height, width]
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


def run_experiment(seed, dims, env_name):
    env = Monitor(BaseEnv(env_name, use_minimal_action_set=True), f'{env_name}/arch_{dims[0]}/seed_{seed}/')

    policy_kwargs = dict(features_extractor_class=MinatarFeaturesExtractor,
                        features_extractor_kwargs=dict(dims=dims),
                        net_arch=[128])

    # DQN hyperparameters
    dqn_kwargs = dict(
        batch_size=32,
        buffer_size=100000,
        target_update_interval=1000,
        train_freq=1,
        learning_starts=5000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        learning_rate=0.00025,
        gamma=0.99,
        verbose=0,
        device='cpu',
        policy_kwargs=policy_kwargs
    )

    model = DQN('MlpPolicy', env, **dqn_kwargs)
    # print(len(model.policy.parameters_to_vector()))
    model.learn(1e7)
    model.save(f'{env_name}/arch_{dims[0]}/seed_{seed}/model')

# Calculate balanced architectures for each environment
# We need to calculate n_flatten for each environment first
def get_balanced_architectures_for_env(env_name, reference_dims=(128, 128)):
    """Get balanced architectures for a specific environment"""
    # Create a temporary environment to calculate n_flatten
    temp_env = BaseEnv(env_name, use_minimal_action_set=True)
    temp_extractor = MinatarFeaturesExtractor(temp_env.observation_space, reference_dims)
    
    # Calculate n_flatten
    with torch.no_grad():
        sample_obs = torch.as_tensor(temp_env.observation_space.sample()[None]).float()
        sample_obs = sample_obs.permute(0, 3, 1, 2)
        n_flatten = temp_extractor.cnn(sample_obs).shape[1]
    
    return get_balanced_architectures(n_flatten, reference_dims)

# Generate balanced architectures for each environment
env_names = ['breakout', 'asterix', 'space_invaders', 'freeway', 'seaquest']
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