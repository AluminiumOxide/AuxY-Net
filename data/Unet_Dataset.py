import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat as load


class Unet_Dataset(Dataset):
    def __init__(self, p0_dir, p0_var_name, p1_dir, p1_var_name):
        super(Unet_Dataset, self).__init__()
        self.p0_dir = p0_dir
        self.p1_dir = p1_dir
        self.p0_var_name = p0_var_name
        self.p1_var_name = p1_var_name
        self.p0_name_list = os.listdir(self.p0_dir)  # 所有p0的mat名称
        self.p1_name_list = os.listdir(self.p1_dir)  # 所有p1的mat名称
        self.transform = get_transform()

    def __getitem__(self, item):
        p0_name = self.p0_name_list[item]  # 获取对应图片名称
        p1_name = self.p1_name_list[item]  # 获取对应图片名称
        p0_name_path = os.path.join(self.p0_dir, p0_name)
        p1_name_path = os.path.join(self.p1_dir, p1_name)
        numpy_p0 = self.get_data_from_mat(p0_name_path, self.p0_var_name, False)
        numpy_p1 = self.get_data_from_mat(p1_name_path, self.p1_var_name, False)

        # Image.fromarray(np.uint8(which_numpy_ndarray_you_need_change_to_Image)).convert('RGB')
        to_uint8_p0 = (numpy_p0 - numpy_p0.min()) / (numpy_p0.max() - numpy_p0.min()) * 255
        tensor_p0 = self.transform(Image.fromarray(np.uint8(to_uint8_p0)).convert('RGB'))
        to_uint8_p1 = (numpy_p1 - numpy_p1.min()) / (numpy_p1.max() - numpy_p1.min()) * 255
        tensor_p1 = self.transform(Image.fromarray(np.uint8(to_uint8_p1)).convert('RGB'))

        return tensor_p0, tensor_p1

    def __len__(self):
        return len(self.p0_name_list)

    @staticmethod
    def get_data_from_mat(mat_path, var_name, show_image=False):
        """
        input a mat file path and variable name you want to fetch from mat file
        :param mat_path: mat file path
        :param var_name: which var you need in mat file
        :param show_image: as param show
        :return: a numpy image with 3*256*356
        """
        input_mat = load(mat_path)
        input_data = input_mat[var_name]
        input_data = np.reshape(input_data, [256, 256, 1])
        output_data = np.concatenate((input_data, input_data, input_data), axis=2)
        if show_image:
            plt.imshow(output_data)
            plt.colorbar()
            plt.show()
            print(type(output_data))
        return output_data


def get_transform():
    transform_list = []
    # transform_list.append(transforms.Resize([286, 286],  interpolation = transforms.InterpolationMode.BILINEAR))
    # transform_list.append(transforms.RandomCrop(256))
    # transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


