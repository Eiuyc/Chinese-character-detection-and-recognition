import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16*16, 1200)
        self.fc2 = nn.Linear(1200, 3755)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 128 -> 128 -> 64
        x = self.pool(F.relu(self.conv2(x))) # 64 -> 64 -> 32
        x = self.pool(F.relu(self.conv3(x))) # 32 -> 32 -> 16
        x = x.view(-1, 32 * 16*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x