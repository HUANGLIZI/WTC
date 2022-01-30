from torch.nn import functional as F
import torch.nn as nn


class AARNet(nn.Module):
    def __init__(self) -> object:
        super(AARNet, self).__init__()
        self.fc1 = nn.Linear(121, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 121)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 11)
        self.dropout = nn.Dropout(0.2)
        # self.relu=F.relu()

    def forward(self, x):
        x = self.dropout(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.fc4(x)
        # x = F.softmax(x,dim=-1)
        return x
