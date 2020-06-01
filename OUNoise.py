# defining the OUNoise process
import numpy as np
import torch
import random

class OUNoise:
    def __init__(self, action_dimension, seed, scale=1., mu=0, theta=0.15, sigma=0.1):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state * self.scale
