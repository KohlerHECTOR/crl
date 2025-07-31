from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from torch import nn
import gymnasium as gym
from typing import Optional, Union
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.distributions import Distribution

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    

class GoalMinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None:
        observation_space = observation_space['observation']
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations['observation']))
    

class GoalMlpExtractor(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: list[list[nn.Module]] = [[] for _ in range(16)]
        value_net: list[list[nn.Module]] =  [[] for _ in range(16)]
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for goal in range(16):
            last_layer_dim_pi = feature_dim

            for curr_layer_dim in pi_layers_dims:
                policy_net[goal].append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
                policy_net[goal].append(activation_fn())
                last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
            last_layer_dim_vf = feature_dim

            for curr_layer_dim in vf_layers_dims:
                value_net[goal].append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
                value_net[goal].append(activation_fn())
                last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module

        self.policy_nets = nn.ModuleList([nn.Sequential(*policy_net[goal]).to(device) for goal in range(16)])
        self.value_nets = nn.ModuleList([nn.Sequential(*value_net[goal]).to(device) for goal in range(16)])

    def forward(self, features: torch.Tensor, goal) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features, goal), self.forward_critic(features, goal)

    def forward_actor(self, features: torch.Tensor, goal) -> torch.Tensor:
        # goal is a batch of integers - flatten if 2D
        if goal.dim() == 2:
            goal = goal.squeeze()  # Remove the last dimension if it's 1
            batch_size = goal.shape[0]
            # Find unique goals to avoid redundant computation
            unique_goals = torch.unique(goal)
            # Initialize output tensor
            output = torch.zeros(batch_size, self.latent_dim_pi, device=features.device)
            
            # Process each unique goal with its corresponding features
            for g in unique_goals:
                # Find indices where goal equals this unique goal
                mask = goal == g  # Ensure mask is 1D
                # Process only the features for this goal
                goal_features = features[mask]
                # Get outputs for this goal
                goal_outputs = self.policy_nets[int(g)](goal_features)
                # Place outputs in correct positions
                output[mask] = goal_outputs
            
            return output
        else:
            # goal is a single integer
            return self.policy_nets[goal](features)

    def forward_critic(self, features: torch.Tensor, goal) -> torch.Tensor:
        # Handle batch of goals efficiently
            # goal is a batch of integers - flatten if 2D
        if goal.dim() == 2:
            goal = goal.squeeze()  # Remove the last dimension if it's 1
            batch_size = goal.shape[0]
            # Find unique goals to avoid redundant computation
            unique_goals = torch.unique(goal)
            
            # Initialize output tensor
            output = torch.zeros(batch_size, self.latent_dim_vf, device=features.device)
            
            # Process each unique goal with its corresponding features
            for g in unique_goals:
                # Find indices where goal equals this unique goal
                mask = goal == g  # Ensure mask is 1D
                # Process only the features for this goal
                goal_features = features[mask]
                # Get outputs for this goal
                goal_outputs = self.value_nets[int(g)](goal_features)
                # Place outputs in correct positions
                output[mask] = goal_outputs
            
            return output
        else:
            # goal is a single integer
            return self.value_nets[goal](features)



class GoalActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = GoalMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features, obs['goal'])
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features, obs['goal'])
            latent_vf = self.mlp_extractor.forward_critic(vf_features, obs['goal'])
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


    def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features, obs['goal'])
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features, obs['goal'])
            latent_vf = self.mlp_extractor.forward_critic(vf_features, obs['goal'])
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features, obs['goal'])
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features, obs['goal'])
        return self.value_net(latent_vf)