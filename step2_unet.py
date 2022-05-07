import torchvision
from data.DP_Dataset import DP_Dataset
from nets.unet import Unet
from nets.DP_Net import DP_Net,DP_Net_new
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


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
    input_P0, test_ua, test_output_ua = img_list
    input_P0 = input_P0.squeeze()
    test_ua = test_ua.squeeze()
    test_output_ua = test_output_ua.squeeze()

    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 3), dpi=240)

    sub = fig.add_subplot(131)
    sub.set_title('input_P0', fontsize=12)
    im = sub.imshow(input_P0)
    sub.tick_params(labelsize=5)
    sub.axis('off')

    sub = fig.add_subplot(132)
    sub.set_title('label_ua', fontsize=12)
    im = sub.imshow(test_ua)
    sub.tick_params(labelsize=5)
    sub.axis('off')

    sub = fig.add_subplot(133)
    sub.set_title('output_ua', fontsize=12)
    im = sub.imshow(test_output_ua)
    sub.tick_params(labelsize=5)
    sub.axis('off')
    cax = add_right_cax(sub, pad=0.035, width=0.02)
    cb = fig.colorbar(im, cax=cax)

    plt.savefig('test_img.png')
    if if_show_img:
        plt.show()
    plt.cla()
    plt.close("all")


if __name__ == '__main__':

    Cuda = True
    dataset_dir = './datasets/cylinder_imitation/'

    train_dir_p0 = dataset_dir + 'train_p0'  # 128_0505
    train_dir_fai = dataset_dir + 'train_fai'
    train_dir_ua = dataset_dir + 'train_ua'
    train_dir_p1 = dataset_dir + 'train_p1'

    val_dir_p0 = dataset_dir + 'val_p0'  # 128_0505
    val_dir_fai = dataset_dir + 'val_fai'
    val_dir_ua = dataset_dir + 'val_ua'
    val_dir_p1 = dataset_dir + 'val_p1'

    batch_size = 4

    train_data = DP_Dataset(train_dir_p0, 'p0', train_dir_p1, 'p0_1', train_dir_ua, 'ua_true', train_dir_fai, 'fai_1')
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_data = DP_Dataset(val_dir_p0, 'p0', val_dir_p1, 'p0_1', val_dir_ua, 'ua_true', val_dir_fai, 'fai_1')
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    print("{} mat images for train and {} mat images for val".format(len(train_data), len(val_data)))
    print('with batch_size {}'.format(batch_size))

    unet = Unet(in_channels=3, pretrained=None)
    # dp_net = DP_Net_new()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        dp_net = unet.cuda()

    # 等下我没看懂原文，就是用均方吗？！
    loss_func_ua = nn.MSELoss()

    lr = 1e-4
    optimizer = optim.Adam(unet.parameters(), lr)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=pow(0.1, 1/10))  # 10次衰减到0.1

    work_dir = './workdir/u_net_cylinder'
    writer = SummaryWriter("{}/logs".format(work_dir))

    epoch = 100
    print("Full dp_net build with unet_1,unet_2,end")
    print("ua  for dp_net.unet_1 \nfai for dp_net.unet_2\nP1  for all dp_net")
    print("Begin train with {} epoch".format(epoch))
    for i in range(epoch):
        print("-------epoch  {} -------".format(i+1))
        # 训练步骤
        unet.train()
        for step, [P0, P1, ua, fai] in enumerate(train_dataloader):
            if torch.cuda.is_available():
                P0 = P0.cuda()
                P1 = P1.cuda()
                ua = ua.cuda()
                fai = fai.cuda()
            # forward
            output_ua = unet(P0)
            loss = loss_func_ua(output_ua, ua)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step = len(train_dataloader) * i + step + 1

            if step % 50 == 0:
                print("step {} train time：{}, Loss unet|ua:{:.4f}".format(
                    step, train_step, loss.item()))

                writer.add_scalar("train_loss/ua", loss.item(), train_step)

                input_P0 = torch.unsqueeze(P0[0].cpu(),dim=0).numpy().mean(axis=1)
                test_ua = torch.unsqueeze(ua[0].cpu(),dim=0).numpy().mean(axis=1)
                test_output_ua = torch.unsqueeze(output_ua[0].detach().cpu(),dim=0).numpy().mean(axis=1)

                img_list = [input_P0, test_ua, test_output_ua]

                draw_img(img_list)

                label_and_output_image = np.array(Image.open('test_img.png'))[:, :, 0:3]

                writer.add_image('input_P0', input_P0, len(train_dataloader)*i//50 + train_step // 50, dataformats='CHW')
                writer.add_image('label_P1', test_ua, len(train_dataloader)*i//50 + train_step // 50, dataformats='CHW')
                writer.add_image('output_P1', test_output_ua, len(train_dataloader)*i//50 + train_step // 50, dataformats='CHW')
                writer.add_image('label_and_output', label_and_output_image, len(train_dataloader)*i//50 + train_step // 50, dataformats='HWC')

        # 测试步骤
        unet.eval()  # 我又没分验证集，怎么测试啊！
        print('--val step--')
        torch.no_grad()  # to increase the validation process uses less memory
        with torch.no_grad():
            unet.eval()  # 我又没分验证集，怎么测试啊！
            total_val_loss = 0
            for step, [P0, P1, ua, fai] in enumerate(val_dataloader):

                if torch.cuda.is_available():
                    P0 = P0.cuda()
                    P1 = P1.cuda()
                    ua = ua.cuda()
                    fai = fai.cuda()
                output_ua = unet(P0)

                loss = loss_func_ua(output_ua, ua)
                total_val_loss += loss.item()
            valid_loss = total_val_loss / len(val_data)
            print("epoch {} val loss {}".format(i, valid_loss))
            writer.add_scalar("val_loss", valid_loss, i)

        # 学习率衰减
        print("learning rate {}".format(lr_scheduler.get_last_lr()[0]))
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()

        if i > 80 or i % 10 == 0:
            torch.save(unet, "{}/module_{}.pth".format(work_dir, i+1))
            print("saved epoch {}".format(i+1))

    writer.close()


