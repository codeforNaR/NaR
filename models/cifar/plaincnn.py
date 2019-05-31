import torch.nn as nn 
import torch.nn.functional as F 

__all__ = ['plaincnn', 'LeNet']

class BasicBlock(nn.Module):
    def __init__(self, nin=3, k=3, nch=64, nlayer=2):
        super(BasicBlock, self).__init__()

        #self.conv = nn.Sequential()
        conv = list()
        padding = (k-1)//2
        for i in range(nlayer):
            conv.append(nn.Conv2d(nin, nch, kernel_size=k, bias=False, stride=1, padding=padding))
            conv.append(nn.BatchNorm2d(nch))
            if i != nlayer - 1:
                conv.append(nn.ReLU())
            nin = nch
        
        self.conv = nn.Sequential(*conv)
    
    def forward(self, x):
        return self.conv(x)
    
class PlainCNN(nn.Module):
    def __init__(self, nchannel=3, depth=6, nb_classes=10):
        super(PlainCNN, self).__init__()

        nlayer = depth//3
        base_ch = 64

        self.block1 = BasicBlock(nchannel, k=3, nch=base_ch, nlayer=nlayer)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = BasicBlock(nin=base_ch, k=3, nch=base_ch*2, nlayer=nlayer)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = BasicBlock(nin=base_ch*2, k=3, nch=base_ch*4, nlayer=nlayer)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, nb_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        o = self.fc2(x)
        
        return o
    
def plaincnn(num_classes=10, depth=6):
    return PlainCNN(nchannel=3, depth=depth, nb_classes=num_classes)


class LeNet(nn.Module):
    def __init__(self, nchannel=3, num_classes=10):
        super(LeNet, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(3, 6, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(6, 16, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out