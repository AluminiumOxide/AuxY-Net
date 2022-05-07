from scipy.io import loadmat as load
import os
import numpy as np
import matplotlib.pyplot as plt


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
    input_data = np.reshape(input_data, [256, 256, 1] )
    output_data = np.concatenate((input_data, input_data, input_data), axis=2)
    if show_image:
        plt.imshow(output_data)
        plt.colorbar()
        plt.show()
        print(type(output_data))
    return output_data

# input_p0 = load(os.path.join(dir_p0, '1.mat'))
# input_p0 = input_p0['p0']
# input_p0 = np.reshape(input_p0, [256, 256, 1])
# input_p0 = np.concatenate((input_p0, input_p0, input_p0), axis=2)
# plt.imshow(input_p0)
# # #plt.imshow(np.reshape(y_test,(128,128)))
# plt.colorbar()
# plt.show()
# print(type(input_p0))


if __name__ == '__main__':
    dir_p0 = "./datasets/brain_and_brain/p0"  # 128_0505
    dir_fai = "./datasets/brain_and_brain/fai"
    dir_ua = "./datasets/brain_and_brain/ua"
    dir_p0_1 = "./datasets/brain_and_brain/p0_1"

    path = os.path.join(dir_fai, '1.mat')
    os.listdir(dir_p0)
    img_p0 = get_data_from_mat(dir_p0+'/1.mat', 'p0', False)
    # img_fai = get_data_from_mat(dir_fai+'/1.mat', 'fai_1', False)
    # img_ua = get_data_from_mat(dir_ua+'/1.mat', 'ua_true', False)
    # img_p0_1 = get_data_from_mat(dir_p0_1+'/1.mat', 'p0_1', False)
    pass
