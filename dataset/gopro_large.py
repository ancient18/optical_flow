import os
from __init__ import DataSet


class GOPRO_Large(DataSet):
    """
    只需实现get_list方法，其他都在基类实现过了
    """

    def __init__(self, args, train=True):
        super(GOPRO_Large, self).__init__(args, train)

    def get_list(self):
        """
        获取图片路径
        :return: [模糊图片路径列表、清晰图片路径列表]
        """
        blur_list = []
        sharp_list = []
        if self.train:
            data_path = os.path.join(self.data_root, 'train')  # D:\dataset\GOPRO_Large\train
        else:
            data_path = os.path.join(self.data_root, 'test')  # D:\dataset\GOPRO_Large\test

        for gopro_num in os.listdir(data_path):
            for png in os.listdir(os.path.join(data_path, str(gopro_num), 'blur')):
                blur_list.append(os.path.join(data_path, str(gopro_num), 'blur', png))
                sharp_list.append(os.path.join(data_path, str(gopro_num), 'sharp', png))

        assert (len(blur_list) == len(sharp_list))
        return [blur_list, sharp_list]
