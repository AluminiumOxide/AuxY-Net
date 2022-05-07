import torchvision
from data.Unet_Dataset import Unet_Dataset
from nets.deeplabv3 import deeplabv3_resnet50
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import skimage.metrics

def convert_to_uint8(input_image):
    # drive numpy image from float32 to uint8
    output_image = np.uint8((input_image - input_image.min()) / (input_image.max() - input_image.min()) * 255)
    return output_image

Cuda = True


dir_p0 = "./datasets/brain_and_brain/p0"  # 128_0505
dir_fai = "./datasets/brain_and_brain/fai"
dir_ua = "./datasets/brain_and_brain/ua"
dir_p0_1 = "./datasets/brain_and_brain/p0_1"

# train_data = Q_PAT(dir_p0, 'p0', dir_p0_1, 'p0_1')
train_data = Unet_Dataset(dir_p0, 'p0', dir_ua, 'ua_true')
train_dataloader = DataLoader(train_data, batch_size=1)


for step, [P0, P1] in enumerate(train_dataloader):
    input_P0 = P0.cpu().numpy().mean(axis=1)
    test_P1 = P1.cpu().numpy().mean(axis=1)
    test_output = P1.cpu().numpy().mean(axis=1)
    input_P0_uint8 = convert_to_uint8(input_P0)
    test_P1_uint8 = convert_to_uint8(test_P1)
    test_output_uint8 = convert_to_uint8(test_output)

    # SSIM值的范围为0至1，越大代表图像越相似。如果两张图片完全一样时，SSIM值为1。
    ssim_score = ssim(test_P1_uint8.squeeze(), input_P0_uint8.squeeze(), data_range=255)
    # PSNR的单位为dB，其值越大，图像失真越少。
    psnr_score = psnr(test_P1_uint8.squeeze(), input_P0_uint8.squeeze(), data_range=255)

    # 这个就不用多说了
    mse_score = mse(test_P1_uint8.squeeze(), input_P0_uint8.squeeze())

    nr_mse_score = skimage.metrics.normalized_root_mse(test_P1_uint8.squeeze(), test_P1_uint8.squeeze())
    # 看起来是越大越好的亚子
    nmi_score = skimage.metrics.normalized_mutual_information(test_P1_uint8.squeeze(), test_P1_uint8.squeeze(), bins=100)
    # img_list = [input_P0, test_P1, test_output]

    break





