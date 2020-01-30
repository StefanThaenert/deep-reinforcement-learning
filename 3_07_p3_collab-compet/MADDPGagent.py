import numpy as np
import random
import copy
from collections import namedtuple, deque

from MADDPGmodel import Actor_Critic_Models
from ReplayBuffer import ReplayBuffer
from DDPGagent import DDPGagent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from param import param

#BUFFER_SIZE = int(1e5)        # reply buffer size
#BATCH_SIZE = 128              # minibatch size
#GAMMA = 0.99                  # discount factor
#TAU = 1e-3                    # for soft update of target parameters
#LR_ACTOR = 1e-4               # learning rate of the actor
#LR_CRITIC = 1e-3              # learning rate of the critic
#WEIGHT_DECAY = 0              # L2 weight decay
#UPDATE_EVERY = 2              # How often to update the network

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device= "cpu"

class maddpgAGENT():
    """maddpgAGENT that contains two DDPG agents and shared replay buffer"""
    def __init__(self):
        """
        Params
        ======
            action_size (int): dimension of each action
            seed (int): Random seed
            n_agents (int): number of distinct agents
            noise_start (float): initial noise weighting factor
            noise_decay (float): noise decay rate
            t_stop_noise (int): max number of timesteps with noise applied in training
        """
        self.Param=param()
        
        #self.buffer_size = BUFFER_SIZE
        #self.batch_size = BATCH_SIZE
        #self.gamma = GAMMA
        #self.tau = TAU
        #self.lr_actor = LR_ACTOR
        #self.lr_critic = LR_CRITIC
        #self.weight_decay = WEIGHT_DECAY
        #self.update_every = UPDATE_EVERY
        #self.n_agents = n_agents
        self.noise_weight = self.Param.noise_start
        #self.noise_decay = noise_decay
        self.t_step = 0
        #self.noise_on = True
        #self.t_stop_noise = t_stop_noise

        # create two agents, each with their own actor and critic
        models = [Actor_Critic_Models() for _ in range(self.Param.num_agents)]
        self.agents = [DDPGagent(i, models[i]) for i in range(self.Param.num_agents)]

        # create shared replay buffer
        self.memory = ReplayBuffer()

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        all_next_states = all_next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)

        # if t_stop_noise time steps are achieved turn off noise
        if self.t_step > self.Param.t_stop_noise:
            self.noise_on = False

        self.t_step = self.t_step + 1
        # Learn every update_every time steps.
        if self.t_step % self.Param.update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.Param.batch_size:
                # sample from the replay buffer for each agent
                experiences = [self.memory.sample() for _ in range(self.Param.num_agents)]
                self.learn(experiences)

    def act(self, all_states, add_noise=True):
        # pass each agent's state from the environment and calculate its action
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state)#, noise_weight=self.noise_weight, add_noise=self.noise_on)
            #self.noise_weight *= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)  # reshape 2x2 into 1x4 dim vector

    def learn(self, experiences):
        # each agent uses its own actor to calculate next_actions
        all_next_actions = []
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            # extract agent i's state and get action via actor network
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)
            # extract agent i's next state and get action via target actor network
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)

        # each agent learns from its experience sample
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], all_next_actions, all_actions)

    def save_agents(self):
        # save models for local actor and critic of each agent
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f"checkpoint_actor_agent_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_agent_{i}.pth")
