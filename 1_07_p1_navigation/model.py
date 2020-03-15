import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=16):
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

    
    
class QNetwork_pixel(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork_pixel, self).__init__()
        nfilters = [128, 128*2, 128*2]
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv3d(3, 128, kernel_size=(1, 3, 3), stride=(1,3,3))
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1,3,3))
        self.bn2 = nn.BatchNorm3d(256)
        self.conv3 = nn.Conv3d(256, 256, kernel_size=(4, 3, 3), stride=(1,3,3))
        self.bn3 = nn.BatchNorm3d(256) # 
        
        
        # Define output size of conv layers
#        y = torch.rand(state_size)
#        y = F.relu(self.bn1(self.conv1(y)))
#        y = F.relu(self.bn2(self.conv2(y)))
#        y = F.relu(self.bn3(self.conv3(y)))
#        y = y.reshape(x.size(0),-1)
#        n_size = y.data.view(1, -1).size(1)
        

        ###############################
        self.fc1=nn.Linear(int(768),int(224))
        self.fc2=nn.Linear(int(224),int(48))
        self.fc3=nn.Linear(int(48),action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x