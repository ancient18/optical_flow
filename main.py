import torch.nn as nn
import torch
import torch.nn.functional as F

from de_conv1 import DeformConv2d1
from backbone import Backbone
from transpose import Transpose
from optical_flow import calcuate_flow
from position_loss import PositionLoss
from vgg import VGG


class MainBranchModule(nn.Module):
    def __init__(self):
        super(MainBranchModule, self).__init__()
        self.backbone = Backbone()
        self.de_conv = DeformConv2d1(128, 128)
        self.transpose = Transpose()

    def forward(self, x):
        img = x
        x = self.backbone(img)
        x, offset, modulation = self.de_conv(x)
        x = self.transpose(x)
        x = img+x
        return x, offset, modulation


class opticalFlowModel(nn.Module):
    def __init__(self):
        super(opticalFlowModel, self).__init__()
        self.optical_flow = calcuate_flow

    def forward(self, im1, im2):
        return self.optical_flow(im1, im2)


net1 = MainBranchModule()
x1 = torch.rand([4, 3, 12, 24])
x1, offset, modulation = net1(x1)
# print(x.shape)
# print(offset.shape)
# print(modulation.shape)


im1 = torch.rand([4, 3, 12, 24])
im2 = torch.rand([4, 3, 12, 24])
net2 = opticalFlowModel()
x2 = net2(im1, im2)
# print(x.shape)


L1 = torch.nn.MSELoss(reduction='sum')
L2 = VGG()
L3 = PositionLoss()
x2 = F.interpolate(x2, scale_factor=0.5, mode='bilinear')
print(x2.shape)
print(offset.shape)
print(L3(offset, x2))
