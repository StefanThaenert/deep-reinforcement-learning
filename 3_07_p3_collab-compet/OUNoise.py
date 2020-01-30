import numpy as np
import random
import copy
from param import param

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, mu=0.):
        """Initialize parameters and noise process."""
        self.Param=param()
        random.seed(self.Param.seed)
        np.random.seed(self.Param.seed)
        self.mu = mu * np.ones(self.Param.action_size)
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.Param.noise_theta * (self.mu - x) + self.Param.noise_sigma * np.random.randn(self.Param.action_size)
        self.state = x + dx
        return self.state

