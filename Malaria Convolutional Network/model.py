import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3,6,5)
        self.pool =  nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,6,5)
        self.fc1 = nn.Linear(6 * 22 * 22,100)
        self.fc2 = nn.Linear(100,1)
        self.sig = nn.Sigmoid()


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,6 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sig(x)

        return x

net = CNN()