from typing import Tuple, Sequence

import numpy as np
import torch
import torch.distributions as distributions

from learner.rl.agent import Agent
from learner.rl.policy import DiscretePolicyNetwork

class ReinforceAgent(Agent):
    """An implementation of the REINFORCE algorithm, mostly as described in the Open AI
    Spinning Up tutorial: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

    Author -- Alex Mitrevski

    """
    def __init__(self, obs_len: int,
                 number_of_actions: int,
                 hidden_layer_size: int = 256,
                 pretrained_model_path: str = None,
                 debug: bool = False):
        super(ReinforceAgent, self).__init__()
        self.policy = DiscretePolicyNetwork(obs_len=obs_len,
                                            number_of_actions=number_of_actions,
                                            hidden_layer_size=hidden_layer_size,
                                            pretrained_model_path=pretrained_model_path)
        self.debug = debug
        self.action_distribution = distributions.Categorical(1. / obs_len * torch.ones(obs_len))

    def sample_action(self, obs: np.array) -> torch.Tensor:
        """Samples an action given an observation.

        Keyword arguments:
        obs: np.array -- a set of observations

        """
        self.action_distribution = self.policy.forward(obs)
        action = self.action_distribution.sample()
        return action

    def get_log_prob(self, action: torch.Tensor) -> float:
        """Returns the log probability of the given action
        under the current action distribution.

        Keyword arguments:
        action: torch.Tensor -- action index

        """
        return self.action_distribution.log_prob(action)

    def update(self, episode_weights: np.ndarray) -> None:
        """Updates the network based on the given episode weights.

        Keyword arguments:
        episode_weights: np.ndarray -- performs a network update based on the given episode
                                       weights, which are given as products of the form
                                       episode_return * episode_length

        """
        self.policy.optimizer.zero_grad()
        loss = -torch.mean(episode_weights)
        loss.backward()
        self.policy.optimizer.step()

    def train(self, env,
              number_of_iterations: int,
              episodes_per_iteration: int,
              render_env: bool = False) -> Sequence[float]:
        """Trains an agent on a given environment.

        Keyword arguments:
        env: an OpenAI gym environment
        number_of_iterations: int -- number of training iterations (i.e. number of times
                                     network updates are performed)
        episodes_per_iteration: int -- number of episodes per iteration (i.e. number of episodes
                                       before a network update is performed)
        render_env: bool -- whether to render the environment (default False)

        Returns:
        average_returns: Sequence[float] -- a list of average episode returns per training update

        """
        average_returns = []
        for i in range(number_of_iterations):
            episode_weights = []
            episode_returns = []

            for _ in range(episodes_per_iteration):
                obs = env.reset()
                R = 0
                episode_step_count = 0
                action_log_probs = []
                episode_done = False
                while not episode_done:
                    if render_env:
                        env.render()
                    action = self.sample_action(obs)
                    (obs, reward, episode_done, _) = env.step(action.item())

                    R += reward
                    episode_step_count += 1
                    action_log_probs.append(self.get_log_prob(action).unsqueeze(0))

                episode_returns.append(R)
                cumulative_episode_return = R * episode_step_count
                episode_weight = torch.sum(torch.cat(action_log_probs) * cumulative_episode_return)
                episode_weights.append(episode_weight.unsqueeze(0))

            average_returns.append(np.mean(episode_returns))
            if self.debug:
                print('Iteration {0}: average return {1}'.format(i+1, average_returns[i]))
            self.update(torch.cat(episode_weights))

        if self.debug:
            print('Training complete')
        return average_returns

    def save_parameters(self, path: str) -> None:
        """Saves the network to a given path.

        Keyword arguments:
        path: str -- location where the parameters are saved

        """
        self.policy.save(path)
