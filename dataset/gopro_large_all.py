"""
返回 模糊、清晰、0帧、2帧
"""
import os
import torch
import torchvision.transforms.functional as F
from PIL import Image
from __init__ import DataSet


class GOPRO_Large_all(DataSet):
    def __init__(self, args, train=True):
        super(GOPRO_Large_all, self).__init__(args=args, train=train)

    def get_list(self):
        first_sharp_frame_dic = {
            'GOPR0372_07_00': 3,
            'GOPR0372_07_01': 2,
            'GOPR0374_11_00': 5,
            'GOPR0374_11_01': 4,
            'GOPR0374_11_02': 3,
            'GOPR0374_11_03': 2,
            'GOPR0378_13_00': 6,
            'GOPR0379_11_00': 5,
            'GOPR0380_11_00': 5,
            'GOPR0384_11_01': 4,
            'GOPR0384_11_02': 3,
            'GOPR0384_11_03': 2,
            'GOPR0384_11_04': 1,
            'GOPR0385_11_00': 5,
            'GOPR0386_11_00': 5,
            'GOPR0477_11_00': 5,
            'GOPR0857_11_00': 5,
            'GOPR0868_11_01': 4,
            'GOPR0868_11_02': 3,
            'GOPR0871_11_01': 4,
            'GOPR0881_11_00': 5,
            'GOPR0884_11_00': 5,

            'GOPR0384_11_00': 5,
            'GOPR0384_11_05': 0,
            'GOPR0385_11_01': 4,
            'GOPR0396_11_00': 5,
            'GOPR0410_11_00': 5,
            'GOPR0854_11_00': 5,
            'GOPR0862_11_00': 5,
            'GOPR0868_11_00': 5,
            'GOPR0869_11_00': 5,
            'GOPR0871_11_00': 5,
            'GOPR0881_11_01': 4
        }
        blur_list = []
        sharp_list = []
        frame0_list = []
        frame1_list = []
        frame2_list = []

        if self.train:
            data_path1 = os.path.join(self.data_root, 'train')
            data_path2 = os.path.join(self.gopro_large_all_data_root, 'train')
        else:
            data_path1 = os.path.join(self.data_root, 'test')
            data_path2 = os.path.join(self.gopro_large_all_data_root, 'test')
        print(data_path1)
        for gopro_num in os.listdir(data_path1):
            for png in os.listdir(os.path.join(data_path1, str(gopro_num), 'blur')):  # 得到模糊、清晰图片
                blur_list.append(os.path.join(
                    data_path1, str(gopro_num), 'blur', png))
                sharp_list.append(os.path.join(
                    data_path1, str(gopro_num), 'sharp', png))
            # gopro large all gopro_num 下的所有图片文件名
            all_list = os.listdir(os.path.join(data_path2, gopro_num))
            n_frames = int(
                len(all_list) / len(os.listdir(os.path.join(data_path1, str(gopro_num), 'blur'))))

            # print("gopro_num:{}\tlen(all_list):{}\tn_frames:{}".format(gopro_num, len(all_list), n_frames))

            # first_sharp_frame = 0  # 当前gopro_num下，第一个清晰图片对应的清晰帧编号
            # for i in range(0, len(all_list)):
            #     frame1_path = os.path.join(data_path2, str(gopro_num), all_list[i])
            #     sharp_path = sharp_list[len(frame1_list)]  # 每隔len(frame1_list)张图片，就是下一个gopro_num的第一个清晰图片
            #     image1 = Image.open(frame1_path).convert('RGB')
            #     sharp_image = Image.open(sharp_path).convert('RGB')
            #     image1 = F.to_tensor(image1)
            #     sharp_image = F.to_tensor(sharp_image)
            #     if torch.equal(image1, sharp_image):  # 此处集群与本地计算结果不一致，因此将本地结果硬编码存入字典
            #         first_sharp_frame = i  # 找到第一个清晰帧位置
            #         break

            first_sharp_frame = first_sharp_frame_dic[str(gopro_num)]
            # print("'{}':{},".format(gopro_num, first_sharp_frame))
            for i in range(0, len(all_list), n_frames):
                frame0_list.append(os.path.join(
                    data_path2, str(gopro_num), all_list[i]))
                frame1_list.append(os.path.join(data_path2, str(
                    gopro_num), all_list[first_sharp_frame]))
                first_sharp_frame = first_sharp_frame + n_frames
                frame2_list.append(os.path.join(data_path2, str(
                    gopro_num), all_list[i + n_frames - 1]))
        # exit(0)
        return [blur_list, sharp_list, frame0_list, frame2_list]


args = {
    "crop_size": [256, 256],
    "gaussian_noise": 0.0,
    "rotation": 0

}
data = GOPRO_Large_all(args)


print(data.get_list())
