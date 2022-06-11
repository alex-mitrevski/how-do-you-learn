from typing import Sequence
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim

class PPOAgent(nn.Module):
    """A simple implementation of Proximal Policy Optimization as described in

    J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov,
    "Proximal Policy Optimization Algorithms", CoRR, vol. abs/1707.06347, 2017.

    For simplicity, the implementation does not use a baseline value.

    Author -- Alex Mitrevski

    """
    def __init__(self, obs_len: int,
                 number_of_actions: int,
                 hidden_layer_size: int = 256,
                 pretrained_model_path: str = None,
                 debug: bool = False):
        super(PPOAgent, self).__init__()
        self.input_layer = nn.Linear(obs_len, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, number_of_actions)
        self.debug = debug
        if pretrained_model_path is not None and pretrained_model_path:
            if self.debug:
                print('Loading saved model {0}'.format(pretrained_model_path))
            self.load_state_dict(torch.load(pretrained_model_path))

        self.action_distribution = distributions.Categorical(1. / obs_len * torch.ones(obs_len))

        self.old_input_layer = copy.deepcopy(self.input_layer)
        self.old_hidden_layer = copy.deepcopy(self.hidden_layer)
        self.old_action_distribution = copy.deepcopy(self.action_distribution)

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def sample_action(self, obs: np.ndarray) -> torch.Tensor:
        """Samples an action given an observation.

        Keyword arguments:
        obs: np.ndarray -- a set of observations

        """
        obs_tensor = torch.Tensor(obs)

        out = torch.tanh(self.old_input_layer(obs_tensor))
        old_action_probabilities = torch.softmax(self.old_hidden_layer(out), dim=0)
        self.old_action_distribution = distributions.Categorical(probs=old_action_probabilities)
        action = self.old_action_distribution.sample()

        out = torch.tanh(self.input_layer(obs_tensor))
        action_probabilities = torch.softmax(self.hidden_layer(out), dim=0)
        self.action_distribution = distributions.Categorical(probs=action_probabilities)

        return action

    def get_prob(self, action: torch.Tensor) -> float:
        """Returns the probability of the given action
        under the current action distribution.

        Keyword arguments:
        action: torch.Tensor -- action index

        """
        return self.action_distribution.probs[action]

    def get_old_prob(self, action: torch.Tensor) -> float:
        """Returns the probability of the given action
        under the old action distribution.

        Keyword arguments:
        action: torch.Tensor -- action index

        """
        return self.old_action_distribution.probs[action]

    def update(self, episode_weights: np.ndarray) -> None:
        """Updates the network based on the given episode weights.

        Keyword arguments:
        episode_weights: np.ndarray -- performs a network update based on the given episode
                                       weights, which are given as products of the form
                                       episode_return * episode_length

        """
        self.old_input_layer = copy.deepcopy(self.input_layer)
        self.old_hidden_layer = copy.deepcopy(self.hidden_layer)

        self.optimizer.zero_grad()
        loss = -torch.mean(episode_weights)
        loss.backward()
        self.optimizer.step()

    def train(self, env,
              number_of_iterations: int,
              episodes_per_iteration: int,
              eps: float,
              render_env: bool = False) -> Sequence[float]:
        """Trains an agent on a given environment using PPO.

        Keyword arguments:
        env: an OpenAI gym environment
        number_of_iterations: int -- number of training iterations (i.e. number of times
                                     network updates are performed)
        episodes_per_iteration: int -- number of episodes per iteration (i.e. number of episodes
                                       before a network update is performed)
        eps: float -- epsilon value to define a probability ratio clipping range
        render_env: bool -- whether to render the environment (default False)

        Returns:
        average_returns: Sequence[float] -- a list of average episode returns per training update

        """
        clip_min_bound = 1 - eps
        clip_max_bound = 1 + eps
        average_returns = []
        for i in range(number_of_iterations):
            episode_weights = []
            episode_returns = []

            for _ in range(episodes_per_iteration):
                obs = env.reset()
                R = 0
                episode_step_count = 0
                action_probs = []
                old_action_probs = []
                episode_done = False
                while not episode_done:
                    if render_env:
                        env.render()
                    action = self.sample_action(obs)
                    (obs, reward, episode_done, _) = env.step(action.item())

                    R += reward
                    episode_step_count += 1
                    action_probs.append(self.get_prob(action).unsqueeze(0))
                    old_action_probs.append(self.get_old_prob(action).unsqueeze(0))

                episode_returns.append(R)
                cumulative_episode_return = R * episode_step_count

                # we calculate the policy probability ratios, namely
                # pi(a|s) / pi_{old}(a|s)
                prob_tensor = torch.cat(action_probs)
                old_prob_tensor = torch.cat(old_action_probs)
                prob_ratios = prob_tensor / old_prob_tensor

                # we find the clipped ratios and calculate the objective
                clipped_ratios = torch.clamp(prob_ratios,
                                             clip_min_bound,
                                             clip_max_bound)
                t = torch.stack((prob_ratios * cumulative_episode_return,
                                 clipped_ratios * cumulative_episode_return), dim=-1)
                objective = torch.min(t, dim=1)

                episode_weight = torch.sum(objective.values)
                episode_weights.append(episode_weight.unsqueeze(0))

            average_returns.append(np.mean(episode_returns))
            if self.debug:
                print('Iteration {0}: average return {1}'.format(i+1, average_returns[i]))
            self.update(torch.cat(episode_weights))

        if self.debug:
            print('Training complete')
        return average_returns
