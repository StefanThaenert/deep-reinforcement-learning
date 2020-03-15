import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    
class CONV(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(PolicyNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size=action_size
        ##############################
        self.conv1=nn.Conv2d(state_size,6,kernel_size=3,stride=2) # 37 x 37
        self.conv2=nn.Conv2d(6,12,kernel_size=3,stride=2)         # 17 x 17 
        self.conv3=nn.Conv2d(12,24,kernel_size=3,stride=2)        # 8 x 8 -->8x8x24=1536
        ###############################
        self.fc1=nn.Linear(int(1536),int(384))
        self.fc2=nn.Linear(int(384),int(96))
        self.fc3=nn.Linear(int(96),action_size)

    def forward(self, state):
        
        x=F.relu(self.conv1(state))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.reshape(x.size(0),-1)

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x),1)
                
        return x