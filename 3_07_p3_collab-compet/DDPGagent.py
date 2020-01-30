from OUNoise import OUNoise
import numpy as np
import random
import copy
from collections import namedtuple, deque

from MADDPGmodel import Actor, Critic
from OUNoise import OUNoise
from param import param


import torch
import torch.nn.functional as F
import torch.optim as optim


class DDPGagent():
    """Interacts with and learns from the environment."""
    def __init__(self, agent_id, model):
        self.Param=param()
        random.seed(self.Param.seed)
        self.id = agent_id

        # Actor models
        self.actor_local = Actor()
        self.actor_target = Actor()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), 
                                          self.Param.actor_lr)

        # Critic models
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           self.Param.critic_lr)

        # Initialize same weights for local and target actor and critic
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        self.noise = OUNoise()

    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def act(self,state):
        """
        Give actions to take in a given state according to the policy
        Params
        ======
        state (numpy array): Input state of the environment
        :return: actions (int)
        """
        state = torch.from_numpy(state).float().to(self.Param.device) # Convert state to torch tensor
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if self.Param.noise:
            action += self.noise.sample()
            
        return np.clip(action, -1, 1)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
                θ_target = τ*θ_local + (1 - τ)*θ_target
                Params
                ======
                    local_model (PyTorch model): weights will be copied from
                    target_model (PyTorch model): weights will be copied to
                    tau (float): interpolation parameter
                """
        tau= self.Param.tau
        for target_param, local_param in zip(target_model.parameters(), 
               local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()

    def learn(self, agent_id, experiences, all_next_actions, all_actions):
        """Update policy and value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            all_next_actions (list): each agent's next_action (as calculated by its actor)
            all_actions (list): each agent's action (as calculated by its actor)
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(self.Param.device)
        actions_next = torch.cat(all_next_actions, dim=1).to(self.Param.device)
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states, actions)
        # q_targets = reward of this timestep + discount * Q(st+1,at+1) from target network
        q_targets = rewards.index_select(1, agent_id) + (self.Param.gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))
        # compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        # minimize loss
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        self.actor_optimizer.zero_grad()
        # detach actions from other agents
        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(self.Param.device)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
