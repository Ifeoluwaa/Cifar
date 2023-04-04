import torch.nn as nn
import torch.nn.functional as F
import torch

#define the CNN architecture

class ConvNet(nn.Module):
  def __init__(self):
      super(ConvNet, self).__init__()
      self.conv1 = nn.Conv2d(3, 100, 5)
      self.pool = nn.MaxPool2d(2,2)
      self.conv2 = nn.Conv2d(100, 200, 5)
    
      self.fc1 = nn.Linear(200*5*5, 500)
      self.fc2 = nn.Linear(500, 250)
      self.fc3 = nn.Linear(250, 10)
    
      self.pool = nn.MaxPool2d(2,2)
    

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 200*5*5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x
  