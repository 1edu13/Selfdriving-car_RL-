import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization of weights to improve training stability.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)  #Set the bias to a constant value
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__() #Inicializates the parent PyTorch module

        # 1. Feature Extractor (CNN Backbone) The eyes
        # Input: (4, 96, 96) - Stacked grayscale frames
        # Output: A flattened vector of spatial features
        self.network = nn.Sequential( #CNN which proceces the images

            #It takes the 4 stacked frames as input and creates 32 feature maps. It uses a large kernel (8x8) and stride
            # (4) to quickly reduce the image size.
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)), # Conv 1
            nn.ReLU(), #Activation function tha introduces non-linearity (allows the network to learn more complex functions)

            #Layers continue to extract more abstract features like curves and road borders, while reducing the image size
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), # Conv 2
            nn.ReLU(),

            #Todo
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), # Conv 3
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the size of the CNN output automatically
        # With input (4, 96, 96), the output dim should be calculated via a dummy pass.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 96, 96)
            output_dim = self.network(dummy_input).shape[1]

        # 2. Critic Head (Value Function) The judge
        # Estimates V(s) - The the estimated "Value" of the current state (how much reward the agent expects to get from here onwards).
        # Input: The output of the CNN feature extractor
        # Output: A single scalar value (V(s))
        # Note: We use a separate network for the value function because it is not directly related to the policy.
        # This allows us to optimize it separately, and to have a different learning rate. (Higher learning rate)
        # (V(s) ~ f(s)) = (V(s) - E[V(s)])^2 + sigma^2 (E[V(s)]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(output_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

        # 3. Actor Head (Policy)
        # Predicts the mean action to take in a given state (a distribution over actions)
        # Input: The output of the CNN feature extractor
        # Outputs the Mean (mu) for the action distribution -> A vector matching the size of the action space (Steering, Gas, Brake)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(output_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
            #Note: The final layer has std=0.01. This scales the initial weights down so the agent starts with very small,
            # centered actions (mostly going straight/doing nothing) rather than erratic movements.
        )

        #Lerneable parameter:
        # We learn the log of the standard deviation (log_std) as a separate parameter.
        # This allows the exploration variance to adjust over time independent of the state. -> By adding noise
        # With this the agent can explore (Hight noise) or exploit (Low noise)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        """Returns the value estimation V(s) for the given observations x."""
        # Normalize input if not done in wrapper: x / 255.0
        return self.critic(self.network(x / 255.0)) #Todo

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

        # Actor: Calculate mean and std for the Normal distribution #Todo What is std
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # Create Normal (Gaussian) distribution
        probs = Normal(action_mean, action_std)

        # If we are training (and passed an old action), we skip this.
        if action is None:
            action = probs.sample()

        #Return:
        # action -> The move to take
        # log_prob -> The probability of taking each action
        # entropy -> A mesure of randomness in the policy (for exploration)
        # value -> The estimated value of the current state
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)