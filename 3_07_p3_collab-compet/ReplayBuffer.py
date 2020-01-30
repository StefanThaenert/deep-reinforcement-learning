import numpy as np
import random
from collections import namedtuple, deque
import torch

from param import param

class ReplayBuffer():
    """Buffer to store experience tuples."""

    def __init__(self):
        self.Param=param()
        random.seed(self.Param.seed)
        np.random.seed(self.Param.seed)
        self.memory = deque(maxlen=self.Param.buffer_size)
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.Param.batch_size)
        
        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(self.Param.device)
        
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).float().to(self.Param.device)
        
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(self.Param.device)
        
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(self.Param.device)
        
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.Param.device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)