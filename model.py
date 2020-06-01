# defining the neural networks underlying the actor / critic pairs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    # Actor model
    def __init__(self, state_size, action_size, seed, fc1, fc2):
        super(Actor, self).__init__()

        #self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)
        self.nonlin = F.relu
        self.seed = torch.manual_seed(seed)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state):
        #h1 = self.nonlin(self.fc1(self.bn(state)))
        h1 = self.nonlin(self.fc1(state))
        h2 = self.nonlin(self.fc2(h1))
        h3 = (self.fc3(h2))
        return F.tanh(h3)

class Critic(nn.Module):
    # Critic model
    def __init__(self, state_size, action_size, seed, fc1, fc2):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1+action_size, fc2)
        self.fc3 = nn.Linear(fc2, 1)
        self.nonlin = F.relu
        self.seed = torch.manual_seed(seed)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state, action):
        # critic network outputs the estimated Q-value
        xs = self.nonlin(self.fc1(state))
        h1 = torch.cat((xs, action), dim=1)
        h2 = self.nonlin(self.fc2(h1))
        h3 = (self.fc3(h2))
        return h3
