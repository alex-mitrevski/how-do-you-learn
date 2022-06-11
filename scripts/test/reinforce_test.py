#!/usr/bin/env python3
import argparse
import numpy as np
import gym

from learner.rl.reinforce import ReinforceAgent
from learner.visualisation.vis_utils import plot_confidence_interval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--env', type=str, required=True,
                        help='Name of an OpenAI gym environment')
    parser.add_argument('-i', '--number_of_iterations', type=int, required=True,
                        help='Number of training iterations')
    parser.add_argument('-e', '--episodes_per_iteration', type=int, required=True,
                        help='Number of episodes per training iteration')
    parser.add_argument('-r', '--training_runs', type=int, required=True,
                        help='Number of times to repeat the training')
    parser.add_argument('-pm', '--pretrained_model_path', type=str, default=None,
                        help='Path to a pretrained model (default None)')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env)
    average_returns_per_run = []
    for i in range(args.training_runs):
        print(f'Starting training run {i+1}')
        agent = ReinforceAgent(obs_len=env.observation_space.shape[0],
                               number_of_actions=env.action_space.n,
                               pretrained_model_path=args.pretrained_model_path,
                               debug=args.debug)
        average_returns = agent.train(env=env,
                                      number_of_iterations=args.number_of_iterations,
                                      episodes_per_iteration=args.episodes_per_iteration)
        average_returns_per_run.append(average_returns)
    env.close()

    average_returns_per_run = np.array(average_returns_per_run)

    # we plot the 95% confidence interval
    plot_confidence_interval(data=average_returns_per_run,
                             std_factor=1.96)
