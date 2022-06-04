import torch.nn as nn
import torch

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, dilation=1),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, dilation=1),
            nn.PReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=1),
            nn.PReLU()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        x = self.layer4(x)
        t1 = x
        x = self.layer5(x) + x
        x = self.layer6(x) + x
        t2 = x
        x = self.layer7(x) + x
        x = self.layer8(x) + x
        t3 = x
        x = self.layer9(x) + x
        x = self.layer10(x) + x
        x = [t1, t2, t3, x]
        x = torch.cat(x, dim=1)
        x = self.layer11(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        return x1, x2
