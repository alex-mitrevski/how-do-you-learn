import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim

class DiscretePolicyNetwork(nn.Module):
    def __init__(self, obs_len: int,
                 number_of_actions: int,
                 hidden_layer_size: int = 256,
                 pretrained_model_path: str = None):
        super(DiscretePolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(obs_len, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, number_of_actions)
        if pretrained_model_path is not None and pretrained_model_path:
            print('Loading saved model {0}'.format(pretrained_model_path))
            self.load_state_dict(torch.load(pretrained_model_path))
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, obs: np.ndarray) -> torch.Tensor:
        obs_tensor = torch.Tensor(obs)
        out = torch.tanh(self.input_layer(obs_tensor))
        action_probabilities = torch.softmax(self.hidden_layer(out), dim=0)
        action_distribution = distributions.Categorical(probs=action_probabilities)
        return action_distribution

    def save(self, path: str) -> None:
        """Saves the network to a given path.

        Keyword arguments:
        path: str -- location where the parameters are saved

        """
        torch.save(self.state_dict(), path)
