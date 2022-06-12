import numpy as np

class Agent(object):
    def sample_action(self, obs: np.ndarray):
        raise NotImplementedError('sample_action is not implemented')

    def train(self):
        raise NotImplementedError('train is not implemented')

    def update(self):
        raise NotImplementedError('update is not implemented')

    def save_parameters(self):
        raise NotImplementedError('save_parameters is not implemented')
