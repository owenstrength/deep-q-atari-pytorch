import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # Input: 4 frames (stacked grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the features after convolutions
        # For Pong with 210x160 input, after processing becomes 22x16
        self.fc1 = nn.Linear(64 * 22 * 16, 512)  # Correct size for Pong
        self.fc2 = nn.Linear(512, action_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)