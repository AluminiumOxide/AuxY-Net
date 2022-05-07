
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat as load
import torchvision.transforms as transforms


class DP_Dataset(Dataset):
    def __init__(self, p0_dir, p0_var_name, p1_dir, p1_var_name, ua_dir, ua_var_name, fai_dir, fai_var_name ):
        super(DP_Dataset, self).__init__()
        self.p0_dir = p0_dir
        self.p1_dir = p1_dir
        self.p0_var_name = p0_var_name
        self.p1_var_name = p1_var_name
        self.ua_dir = ua_dir
        self.fi_dir = fai_dir
        self.ua_var_name = ua_var_name
        self.fi_var_name = fai_var_name
        self.p0_name_list = os.listdir(self.p0_dir)  # 所有p0的mat名称
        self.p1_name_list = os.listdir(self.p1_dir)  # 所有p1的mat名称
        self.ua_name_list = os.listdir(self.ua_dir)  # 所有ua的mat名称
        self.fi_name_list = os.listdir(self.fi_dir)  # 所有fai的mat名称
        self.transform = get_transform()

    def __getitem__(self, item):
        p0_name = self.p0_name_list[item]  # 获取对应图片名称
        p1_name = self.p1_name_list[item]  # 获取对应图片名称
        ua_name = self.ua_name_list[item]  # 获取对应图片名称
        fi_name = self.fi_name_list[item]  # 获取对应图片名称
        p0_name_path = os.path.join(self.p0_dir, p0_name)
        p1_name_path = os.path.join(self.p1_dir, p1_name)
        ua_name_path = os.path.join(self.ua_dir, ua_name)
        fi_name_path = os.path.join(self.fi_dir, fi_name)

        # numpy_p0 = self.get_data_from_mat(p0_name_path, self.p0_var_name, False) * 255
        # numpy_p1 = self.get_data_from_mat(p1_name_path, self.p1_var_name, False) * 255
        # numpy_ua = self.get_data_from_mat(ua_name_path, self.ua_var_name, False) * 255
        # numpy_fi = self.get_data_from_mat(fi_name_path, self.fi_var_name, False) * 255
        numpy_p0 = self.get_data_from_mat(p0_name_path, self.p0_var_name, False)
        numpy_p1 = self.get_data_from_mat(p1_name_path, self.p1_var_name, False)
        numpy_ua = self.get_data_from_mat(ua_name_path, self.ua_var_name, False)
        numpy_fi = self.get_data_from_mat(fi_name_path, self.fi_var_name, False)

        # tensor_p0 = torch.from_numpy(numpy_p0.transpose(2, 0, 1)).type(torch.FloatTensor)
        # tensor_p1 = torch.from_numpy(numpy_p1.transpose(2, 0, 1)).type(torch.FloatTensor)
        # tensor_ua = torch.from_numpy(numpy_ua.transpose(2, 0, 1)).type(torch.FloatTensor)
        # tensor_fi = torch.from_numpy(numpy_fi.transpose(2, 0, 1)).type(torch.FloatTensor)
        #
        # normal_p0 = (tensor_p0 - tensor_p0.min()) / (tensor_p0.max() - tensor_p0.min())
        # normal_p1 = (tensor_p1 - tensor_p1.min()) / (tensor_p1.max() - tensor_p1.min())
        # normal_ua = (tensor_ua - tensor_ua.min()) / (tensor_ua.max() - tensor_ua.min())
        # normal_fi = (tensor_fi - tensor_fi.min()) / (tensor_fi.max() - tensor_fi.min())

        to_uint8_p0 = (numpy_p0 - numpy_p0.min()) / (numpy_p0.max() - numpy_p0.min()) * 255
        tensor_p0 = self.transform(Image.fromarray(np.uint8(to_uint8_p0)).convert('RGB'))
        to_uint8_p1 = (numpy_p1 - numpy_p1.min()) / (numpy_p1.max() - numpy_p1.min()) * 255
        tensor_p1 = self.transform(Image.fromarray(np.uint8(to_uint8_p1)).convert('RGB'))
        to_uint8_ua = (numpy_ua - numpy_ua.min()) / (numpy_ua.max() - numpy_ua.min()) * 255
        tensor_ua = self.transform(Image.fromarray(np.uint8(to_uint8_ua)).convert('RGB'))
        to_uint8_fi = (numpy_fi - numpy_fi.min()) / (numpy_fi.max() - numpy_fi.min()) * 255
        tensor_fi = self.transform(Image.fromarray(np.uint8(to_uint8_fi)).convert('RGB'))


        return tensor_p0, tensor_p1, tensor_ua, tensor_fi
        # return normal_p0, normal_p1, normal_ua, normal_fi
        # return numpy_p0, numpy_p1, numpy_ua ,numpy_fi


    def __len__(self):
        return len(self.p0_name_list)

    @staticmethod
    def get_data_from_mat(mat_path, var_name, show_image=False):
        """
        input a mat file path and variable name you want to fetch from mat file
        :param mat_path: mat file path
        :param var_name: which var you need in mat file
        :param show_image: as param show
        # :return: a numpy image with 3*256*356
        :return: a numpy image with 1*256*356
        """
        input_mat = load(mat_path)
        input_data = input_mat[var_name]
        input_data = np.reshape(input_data, [256, 256, 1])
        output_data = np.concatenate((input_data, input_data, input_data), axis=2)
        # output_data = input_data
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
    # transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


if __name__ == '__main__':
    dir_p0 = "../datasets/brain_and_brain/p0"  # 128_0505
    dir_fai = "../datasets/brain_and_brain/fai"
    dir_ua = "../datasets/brain_and_brain/ua"
    dir_p0_1 = "../datasets/brain_and_brain/p0_1"
    train_data = DP_Dataset(dir_p0, 'p0', dir_p0_1, 'p0_1', dir_ua, 'ua_true', dir_fai, 'fai_1')

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_data, batch_size=1)

    for step, [P0, P1, ua, fai] in enumerate(train_dataloader):
        input_P0 = P0.numpy().mean(axis=1)
        test_P1 = P1.numpy().mean(axis=1)
        test_ua = ua.numpy().mean(axis=1)
        test_fi = fai.numpy().mean(axis=1)

        plt.figure()
        plt.subplot(2, 2, 1)
        # plt.colorbar()
        plt.title('input_P0', fontsize=12)
        plt.imshow(input_P0.squeeze())
        plt.tick_params(labelsize=5)
        plt.subplot(2, 2, 2)
        plt.title('test_P1', fontsize=12)
        plt.imshow(test_P1.squeeze())
        plt.tick_params(labelsize=5)
        plt.subplot(2, 2, 3)
        plt.title('test_ua', fontsize=12)
        plt.imshow(test_ua.squeeze())
        plt.tick_params(labelsize=5)
        plt.subplot(2, 2, 4)
        plt.title('test_fi', fontsize=12)
        plt.imshow(test_fi.squeeze())
        plt.tick_params(labelsize=5)

        plt.savefig('latest_img.png')
        plt.cla()
        plt.close("all")