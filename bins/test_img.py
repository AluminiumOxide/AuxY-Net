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


def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


def draw_img(img_list, if_show_img=False):
    input_P0, test_P1, test_output = img_list
    input_P0 = input_P0.squeeze()
    test_P1 = test_P1.squeeze()
    test_output = test_output.squeeze()
    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 3), dpi=240)

    sub = fig.add_subplot(131)
    sub.set_title('input_P0', fontsize=12)
    im = sub.imshow(input_P0)
    sub.tick_params(labelsize=5)

    sub = fig.add_subplot(132)
    sub.set_title('label_ua', fontsize=12)
    im = sub.imshow(test_P1)
    sub.tick_params(labelsize=5)

    sub = fig.add_subplot(133)
    sub.set_title('output_ua', fontsize=12)
    im = sub.imshow(input_P0)
    sub.tick_params(labelsize=5)
    cax = add_right_cax(sub, pad=0.02, width=0.02)
    cb = fig.colorbar(im, cax=cax)

    plt.savefig('test_img.png')
    if if_show_img:
        plt.show()
    plt.cla()
    plt.close("all")

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
    img_list = [input_P0, test_P1, test_output]
    draw_img(img_list)
    break
