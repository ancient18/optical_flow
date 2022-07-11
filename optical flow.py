import cv2
import numpy as np
from torchvision import transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)


def tensor_numpy(image):
    # 去掉batch通道 (batch, C, H, W) --> (C, H, W)
    clean = image.clone().detach().cpu().squeeze(0)
    clean[0] = clean[0] * std[0] + mean[0]                 # 数据去归一化
    clean[1] = clean[1] * std[1] + mean[1]
    clean[2] = clean[2].mul(std[2]) + mean[2]
    # 转换到颜色255 [0, 1] --> [0, 255]
    clean = np.around(clean.mul(255))
    # 跟换三通道 (C, H, W) --> (H, W, C)
    clean = np.uint8(clean).transpose(1, 2, 0)
    r, g, b = cv2.split(clean)                             # RGB 通道转换
    clean = cv2.merge([b, g, r])
    return clean


tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def calcuate_flow(im1, im2):
    im1 = tensor_numpy(im1)
    im2 = tensor_numpy(im2)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(1)
    flow = dis.calc(gray1, gray2, None,)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(im1)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    print(hsv.shape)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('result',bgr)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return tfms(hsv)


im1 = tfms(cv2.imread('000105.png'))
im2 = tfms(cv2.imread('000106.png'))

print(calcuate_flow(im1, im2).shape)

# 网络一：输入模糊图像，输出1.清晰图像，2.偏移量，3.权重

# 网络二：输入两帧图像，输出光流
