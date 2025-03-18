import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        
        # Convolutional layers (expects input shape: 4 x 84 x 84)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute size of the flattened feature maps dynamically
        self._conv_output_size = self._get_conv_output((4, 84, 84))

        # Fully connected layers
        self.fc1 = nn.Linear(self._conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_size)

    def _get_conv_output(self, shape):
        """Pass a dummy tensor to calculate flattened size dynamically."""
        with torch.no_grad():
            x = torch.zeros(1, *shape)  # Batch size of 1
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.numel()  # Total flattened features

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        return self.fc2(x)  # Output Q-values for all actions