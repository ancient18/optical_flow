import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, dilation=1, padding=1),
            nn.PReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=2)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4, padding=4)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4, padding=4)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3,
                      stride=1, dilation=1, padding=1),
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
        x = self.layer10(x)
        x = [t1, t2, t3, x]
        x = torch.cat(x, dim=0)
        x = self.layer11(x)
        print(x.shape)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        return x1, x2


img = cv2.imread("./car.png")


tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR-->RGB
img = tfms(img)  # 转化为tensor 并 归一化
print(img.shape)
a = Backbone()
a(img)
