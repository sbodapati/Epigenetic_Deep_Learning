import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, numFeatures):
        super(Net, self).__init__()
        self.fl1 = nn.Linear(numFeatures, 200)
        self.fl2 = nn.Linear(200, 200)
        self.fl3 = nn.Linear(200, 200)
        self.fl4 = nn.Linear(200, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x.float()
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        x = F.relu(self.fl3(x))
        x = self.fl4(x)
        return x






