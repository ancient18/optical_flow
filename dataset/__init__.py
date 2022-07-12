import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F


class DataSet(data.Dataset):

    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.data_root = 'D:\GOPRO_Large'  # gopro or reds
        self.gopro_large_all_data_root = 'D:\GOPRO_Large_all'  # gopro large all
        self.crop_size = args['crop_size']
        self.gaussian_noise = args['gaussian_noise']
        self.rotation = args['rotation']
        self.image_list = self.get_list()

    def add_noise(self, blur_image):
        """
        训练数据的模糊图像以一定概率添加高斯噪声
        :param blur_image: Image格式
        :return:添加过高斯噪声的Image格式
        """
        rate = np.random.random()
        if rate < self.gaussian_noise:
            blur_image = np.array(blur_image)
            blur_image = blur_image + np.random.normal(0, 0.01 * 255, blur_image.shape)
            blur_image = np.clip(blur_image, 0, 255).astype('uint8')
            blur_image = Image.fromarray(blur_image).convert('RGB')
            # print("已添加高斯噪声")
        return blur_image

    def add_rotation(self, images):
        """
        训练数据以一定概率随机旋转，旋转角度在[90, 180, 270]中随机选择
        :param images: [Image]
        :return: [Image]
        """
        rate = np.random.random()
        if rate < self.rotation:
            degree = np.random.choice([90, 180, 270])
            for i in range(len(images)):
                images[i] = images[i].rotate(degree)
        return images

    def add_crop(self, images):
        """
        训练数据随机剪裁为指定大小
        :param images: [Tensor]
        :return: [Tensor]
        """
        h = images[0].size(1)
        w = images[0].size(2)
        hs = np.random.randint(0, h - self.crop_size + 1, 1)[0]
        ws = np.random.randint(0, w - self.crop_size + 1, 1)[0]
        for i in range(len(images)):
            images[i] = images[i][:, hs:hs + self.crop_size, ws:ws + self.crop_size]
        # print("h:{},w:{},已剪裁hs:{},ws:{}".format(h, w, hs, ws))
        return images

    def get_list(self):
        """
        每个数据集的目录组成都不同，逻辑需要单独写
        一般来说image_list[0]是模糊图片路径列表,image_list[1]是清晰图片路径列表
        或者image_list[0]是模糊图片路径列表,image_list[1]是清晰图片也是第1帧路径列表,image_list[2]是第0帧图片路径列表,image_list[3]是第2帧路径列表
        :return:
        """
        image_list = []
        return image_list

    def __getitem__(self, item):
        """
        读取图片，对于训练集，先剪裁再旋转，然后对模糊图片加噪声，既避免了先旋转后剪裁的bug，又降低CPU与显卡占用
        :param item: 图片编号
        :return: 模糊图片、清晰图片、模糊图片路径、清晰图片路径
        或者 模糊图片、清晰图片也是第1帧、第0帧、第2帧、模糊图片路径、清晰图片也是第1帧路径、第0帧路径、第2帧路径
        总之前一半是图片，后一半是图片路径
        """
        images = []
        for i in range(len(self.image_list)):  # 读取图片
            image = Image.open(self.image_list[i][item]).convert('RGB')
            image = F.to_tensor(image)
            images.append(image)
        if self.train:
            images = self.add_crop(images)  # 随机剪裁 输入[Tensor]
            for i in range(len(images)):
                images[i] = F.to_pil_image(images[i])  # 转为PIL
            images = self.add_rotation(images)  # 旋转 输入[PIL]
            images[0] = self.add_noise(images[0])  # 只对模糊图片添加噪声 输入PIL
            for i in range(len(images)):
                images[i] = F.to_tensor(images[i])  # 转回Tensor
        for i in range(len(self.image_list)):
            images.append(self.image_list[i][item])

        return images

    def __len__(self):
        """
        获取数据集图片数量
        :return:
        """
        return len(self.image_list[0])
