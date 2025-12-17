import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization of weights to improve training stability.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()

        # 1. Feature Extractor (CNN Backbone)
        # Input: (4, 96, 96) - Stacked grayscale frames
        # Output: A flattened vector of spatial features
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)), # Conv 1
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), # Conv 2
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), # Conv 3
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the size of the CNN output automatically
        # With input (4, 96, 96), the output dim should be calculated via a dummy pass.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 96, 96)
            output_dim = self.network(dummy_input).shape[1]

        # 2. Critic Head (Value Function)
        # Estimates V(s) - The value of the state
        self.critic = nn.Sequential(
            layer_init(nn.Linear(output_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

        # 3. Actor Head (Policy)
        # Outputs the Mean (mu) for the action distribution
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(output_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
        )

        # We learn the log of the standard deviation (log_std) as a separate parameter.
        # This allows the exploration variance to adjust over time independent of the state.
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        """Returns the value estimation V(s) for the given observations x."""
        # Normalize input if not done in wrapper: x / 255.0
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        """
        Given observation x, returns:
        - action: The sampled action
        - log_prob: Log probability of that action (for the loss function)
        - entropy: Entropy of the distribution (for exploration bonus)
        - value: Estimated value of the state
        """
        # Note: Assumes x is (Batch, 4, 96, 96)
        # Normalization happening here (or in wrapper)
        hidden = self.network(x / 255.0)

        # Actor: Calculate mean and std for the Normal distribution
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # Create Normal (Gaussian) distribution
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        # Return action, log_prob, entropy, and value
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)