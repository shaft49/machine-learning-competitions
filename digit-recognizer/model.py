import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size2, 64)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(64, num_classes)

    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        return out

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=1) # n 32 26 26
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=1) # n 32 24 24
        self.pool = nn.MaxPool2d(2, 2) # n 32 12 12
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, padding=1) # n 16 10 10
        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 1, 28, 28
        x = F.relu(self.conv1(x))  # -> n, 32, 24, 24
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 32, 12, 12
        x = F.relu(self.conv3(x)) # -> n, 16 5 5
        x = x.view(-1, 16 * 10 * 10)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x
