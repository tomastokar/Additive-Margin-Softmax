import torch.nn as nn
import torch.nn.functional as F 
from AMSloss import AdMSoftmaxLoss

class AMLConv(nn.Module):
    def __init__(self, no_classes = 10):
        super(AMLConv, self).__init__()    

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels = 32, kernel_size = 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.lin1 = nn.Linear(128, 3)

        self.loss = AdMSoftmaxLoss(
            embedding_dim = 3, 
            no_classes = no_classes, 
            scale = 10.0, 
            margin = 0.4
        )


    def forward(self, x, labels = None):
        x = self.conv1(x)        
        x = self.conv2(x)        
        x = self.conv3(x)        
        x = x.reshape(x.size(0), -1)        
        x = self.lin1(x)
        
        if labels is not None:
            return self.loss(x, labels)
        else:
            return x 
        