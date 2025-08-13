import minatar
from minatar import Environment
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from torch import nn
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import obs_as_tensor

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
        self.feats_ = self.linear(self.cnn(observations))
        return self.feats_


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


env = BaseEnv('freeway', use_minimal_action_set=True, render_mode='rgb_array')
policy_kwargs = dict(features_extractor_class=MinatarFeaturesExtractor,
                        features_extractor_kwargs=dict(dims=(128, 128)),
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
model.policy.load_state_dict(torch.load('policy_freeway_128.pth', weights_only=True))
s, _ = env.reset()

# Set up the matplotlib figure for real-time plotting
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Freeway Environment with Feature Activations', fontsize=16)

# Initialize the bar plot
feature_dim = 128
x_pos = np.arange(feature_dim)
bars = ax2.bar(x_pos, np.zeros(feature_dim))
ax2.set_title('Feature Activations')
ax2.set_xlabel('Feature Index')
ax2.set_ylabel('Activation Value')
ax2.set_ylim(0, 10)  # Show range between 0 and 10

# Initialize the environment display
ax1.set_title('Environment State')
ax1.axis('off')

done = False
r_sum = 0
step_count = 0

while not done:
    a = model.predict(s)[0]
    features = model.policy.q_net.features_extractor.feats_.detach().numpy()[0]
    
    # Update the bar plot
    for bar, height in zip(bars, features):
        bar.set_height(height)
    
    # Update the environment display
    env_state = env.render()
    if env_state is not None:
        ax1.clear()
        ax1.imshow(env_state)
        ax1.set_title(f'Environment State - Step {step_count}')
        ax1.axis('off')
    
    # Update the plot
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    s, r, term, trunc, _ = env.step(a)
    r_sum += r
    done = term or trunc
    step_count += 1
    
    # Add a small delay to make the visualization visible
    plt.pause(0.1)

print(f"Total reward: {r_sum}")
plt.ioff()  # Turn off interactive mode
plt.show()