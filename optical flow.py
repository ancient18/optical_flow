import cv2
import numpy as np


def calcuate_flow(im1, im2):
    gray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(1)
    flow = dis.calc(gray1, gray2, None,)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros_like(im1)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # print(hsv.shape)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('result',bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return hsv

# im1 = cv2.imread('000105.png')
# im2 = cv2.imread('000106.png')
# calcuate_flow(im1,im2)