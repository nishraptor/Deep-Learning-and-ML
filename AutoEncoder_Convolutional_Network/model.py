import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        #Encoder Network
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 14, kernel_size=5)
        self.reduced = nn.Linear(14 * 4 * 4,10)

        #Decoder Network
        self.fc1 = nn.Linear(10,100)
        self.fc2 = nn.Linear(100,28 * 28 )

    def encode(self,images):

        code = self.pool(F.relu(self.conv1(images)))
        code = self.pool(F.relu(self.conv2(code)))
        code = code.view(-1, 14 * 4 * 4)
        code = self.reduced(code)

        return code

    def decode(self,rep):
        images = F.relu(self.fc1(rep))
        images = F.sigmoid(self.fc2(images))
        images = images.view([rep.size(0),1,28,28])

        return images

    def forward(self, x):
        code = self.encode(x)
        out = self.decode(code)

        return out,code


net = AutoEncoder()