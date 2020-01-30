import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from param import param

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """"Policy model i.e. model mapping states to actions"""
    def __init__(self):
        self.Param=param()
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(self.Param.seed)
        self.fc1 = nn.Linear(self.Param.state_size, self.Param.actor_layers[0])
        self.fc2 = nn.Linear(self.Param.actor_layers[0], self.Param.actor_layers[1])
        self.fc3 = nn.Linear(self.Param.actor_layers[1], self.Param.action_size)
        self.bn = nn.BatchNorm1d(self.Param.actor_layers[0])
        self.reset_parameters()

    def forward(self, state):
        """Actor network mapping states to actions"""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    def reset_parameters(self):
        """"Initialize weights with non-zero values"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

class Critic(nn.Module):
    """Model estimating the Q-values or action-value function"""
    def __init__(self):
        self.Param=param()
        super(Critic, self).__init__()
        critic_input_size = (self.Param.state_size + self.Param.action_size) * self.Param.num_agents
        self.seed = torch.manual_seed(self.Param.seed)
        self.fc1 = nn.Linear(critic_input_size,self.Param.critic_layers[0])
        self.fc2 = nn.Linear(self.Param.critic_layers[0], self.Param.critic_layers[1])
        self.fc3 = nn.Linear(self.Param.critic_layers[1], 1)
        self.bn = nn.BatchNorm1d(self.Param.critic_layers[0])
        self.reset_parameters()

    def reset_parameters(self):
        """"Initialize weights with non-zero values"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Critic network mapping states to action values"""
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(xs))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation function here as we need to estimate the Q-value
        return x

class Actor_Critic_Models():
    """"Class containing all the models for a DDPG agent"""
    def __init__(self):
        self.Param=param()
        self.actor_local = Actor().to(self.Param.device)
        
        self.actor_target = Actor().to(self.Param.device)
        
        self.critic_local = Critic().to(self.Param.device)
        
        self.critic_target = Critic().to(self.Param.device)