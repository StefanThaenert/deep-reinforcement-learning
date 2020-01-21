import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BatchSize = 128
BufferSize = int(1e5)
Gamma = 0.99
LrActor = 1e-3
LrCritic = 1e-3
Tau = 1e-3
NoiseDecay = 0.999
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        print('DDPG Agent hyperparameters:\n\t' \
              'batch size: \t\t{:8.0f}\n\t' \
              'buffer size: \t\t{:8.0f}\n\t' \
              'discount / gamma: \t{:8.3f}\n\t' \
              'actor learning rate: \t{:8.3f}\n\t' \
              'critic learning rate: \t{:8.3f}\n\t' \
              'soft update / tau: \t{:8.3f}\n\t' \
              'noise decay rate: \t{:8.3f}\n'
              .format(BatchSize, BufferSize, Gamma,
                      LrActor, LrCritic, Tau, NoiseDecay))
        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed, [400, 300]).to(Device)
        self.actor_target = Actor(state_size, action_size, random_seed, [400, 300]).to(Device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LrActor)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed, [400, 300]).to(Device)
        self.critic_target = Critic(state_size, action_size, random_seed, [400, 300]).to(Device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LrCritic)

        # Initialize target networks weights with the local networks ones
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(random_seed)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_decay = NoiseDecay

    def act(self, state, noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(Device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()
        self.actor_local.train()

        if noise:
            # Add noise to the action in order to explore the environment
            action += self.noise_decay * self.noise.sample()
            # Decay the noise process along the time
            self.noise_decay *= self.noise_decay

        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        # Save experience
        self.replay_buffer.add(states, actions, rewards, next_states)

        # Learn, if enough samples are available in memory
        if len(self.replay_buffer) > BatchSize:
            experiences = self.replay_buffer.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
        """
        states, actions, rewards, next_states = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions from actor_target model
        actions_next = self.actor_target(next_states)
        # Get predicted next-state Q-Values from critic_target model
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (Gamma * Q_targets_next)
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
    def soft_update(self, local_model, target_model, tau=Tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            seed (int): random seed
        """
        self.memory = deque(maxlen=BufferSize)
        self.experience = namedtuple('Experience', field_names=['states', 'actions', 'rewards', 'next_states'])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=BatchSize)
        
        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(Device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(Device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(Device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(Device)

        return (states, actions, rewards, next_states)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state