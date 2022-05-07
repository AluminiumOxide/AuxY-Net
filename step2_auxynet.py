import torchvision
from data.DP_Dataset import DP_Dataset
from nets.unet import Unet
from nets.ynet import Ynet
from nets.DP_Net import DP_Net_new
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
    input_P0, test_P1, test_ua, test_fi, test_output_P1, test_output_ua, test_output_fi = img_list
    input_P0 = input_P0.squeeze()
    test_P1 = test_P1.squeeze()
    test_ua = test_ua.squeeze()
    test_fi = test_fi.squeeze()
    test_output_P1 = test_output_P1.squeeze()
    test_output_ua = test_output_ua.squeeze()
    test_output_fi = test_output_fi.squeeze()

    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 4), dpi=240)

    sub = fig.add_subplot(244)
    sub.set_title('input_P0', fontsize=12)
    im = sub.imshow(input_P0)
    sub.tick_params(labelsize=5)
    sub.axis('off')

    sub = fig.add_subplot(241)
    sub.set_title('label_P1', fontsize=12)
    im = sub.imshow(test_P1)
    sub.tick_params(labelsize=5)
    sub.axis('off')

    sub = fig.add_subplot(242)
    sub.set_title('label_ua', fontsize=12)
    im = sub.imshow(test_ua)
    sub.tick_params(labelsize=5)
    sub.axis('off')

    sub = fig.add_subplot(243)
    sub.set_title('label_fai', fontsize=12)
    im = sub.imshow(test_fi)
    sub.tick_params(labelsize=5)
    sub.axis('off')

    sub = fig.add_subplot(245)
    sub.set_title('output_P1', fontsize=12)
    im = sub.imshow(test_output_P1)
    sub.tick_params(labelsize=5)
    sub.axis('off')

    sub = fig.add_subplot(246)
    sub.set_title('output_ua', fontsize=12)
    im = sub.imshow(test_output_ua)
    sub.tick_params(labelsize=5)
    sub.axis('off')

    sub = fig.add_subplot(247)
    sub.set_title('output_fai', fontsize=12)
    im = sub.imshow(test_output_fi)
    sub.tick_params(labelsize=5)
    sub.axis('off')
    cax = add_right_cax(sub, pad=0.035, width=0.02)
    cb = fig.colorbar(im, cax=cax)
    # fig.xticks([])
    # fig.yticks([])
    # plt.axis('off')

    plt.savefig('test_img.png')
    if if_show_img:
        plt.show()
    plt.cla()
    plt.close("all")


if __name__ == '__main__':
    Cuda = True
    batch_size = 4

    dataset_dir = './datasets/mouse/'

    train_dir_p0 = dataset_dir + 'train_p0'  # 128_0505
    train_dir_fai = dataset_dir + 'train_fai'
    train_dir_ua = dataset_dir + 'train_ua'
    train_dir_p1 = dataset_dir + 'train_p1'

    val_dir_p0 = dataset_dir + 'val_p0'  # 128_0505
    val_dir_fai = dataset_dir + 'val_fai'
    val_dir_ua = dataset_dir + 'val_ua'
    val_dir_p1 = dataset_dir + 'val_p1'

    train_data = DP_Dataset(train_dir_p0, 'p0', train_dir_p1, 'p0_1', train_dir_ua, 'ua_true', train_dir_fai, 'fai_1')
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_data = DP_Dataset(val_dir_p0, 'p0', val_dir_p1, 'p0_1', val_dir_ua, 'ua_true', val_dir_fai, 'fai_1')
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    print("{} mat images for train and {} mat images for val".format(len(train_data), len(val_data)))
    print('with batch_size {}'.format(batch_size))

    # unet = Unet(in_channels=3, pretrained=None)
    ynet = Ynet()
    if torch.cuda.is_available():
        # import torch.backends.cudnn as cudnn
        # net = nn.DataParallel(module)  # 我又不用多GPU
        torch.backends.cudnn.benchmark = True
        ynet = ynet.cuda()

    # 等下我没看懂原文，就是用均方吗？！
    loss_func_ua = nn.MSELoss()
    loss_func_fi = nn.MSELoss()
    loss_func_p1 = nn.MSELoss()

    lr = 1e-4
    optimizer = optim.Adam(ynet.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=pow(0.1, 1/10))  # 10次衰减到0.1

    work_dir = './workdir/ynet_mouse'
    writer = SummaryWriter("{}/logs".format(work_dir))

    epoch = 100
    print("Full ynet build with backbone,up_sample_ua,up_sample_ua,end")
    print("ua  for backbone and up_sample_ua \nfai for backbone and up_sample_fi \nP1  for all ynet")
    print("Begin train with {} epoch".format(epoch))
    for i in range(epoch):
        print("-------epoch  {} -------".format(i+1))
        # 训练步骤
        ynet.train()
        for step, [P0, P1, ua, fai] in enumerate(train_dataloader):
            # print("iter {}".format(step))
            # print("P0 info {} {} {}".format(P0.size(),P0.max(),P0.min()) )
            # print("P1 info {} {} {}".format(P1.size(),P1.max(),P1.min()) )
            if torch.cuda.is_available():
                P0 = P0.cuda()
                P1 = P1.cuda()
                ua = ua.cuda()
                fai = fai.cuda()
            # forward
            output_ua, output_fi, output_p1 = ynet(P0)
            P1 = (ua + 1) * (fai + 1) * 0.5 -1  # P1 = ua.data * fai.data     # decay = (ua + 1) * (fai + 1) * 0.5  - P1 -1
            loss_P1 = loss_func_p1(output_p1, P1)
            loss_ua = loss_func_ua(output_ua, ua)
            loss_fi = loss_func_fi(output_fi, fai)
            loss = loss_P1 * 100 + loss_ua * 200 + loss_fi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step = len(train_dataloader) * i + step + 1

            if step % 50 == 0:
                print("step {} train time：{}, Loss up_sample_ua:{:.4f}, up_sample_fai:{:.4f}, all|P1:{:.4f}".format(
                    step, train_step, loss_ua.item(), loss_fi.item(), loss_P1.item()))

                writer.add_scalar("train_loss/ua", loss_ua.item(), train_step)
                writer.add_scalar("train_loss/fi", loss_fi.item(), train_step)
                writer.add_scalar("train_loss/p1", loss_P1.item(), train_step)
                writer.add_scalar("train_loss/mse", loss.item() / 301, train_step)

                input_P0 = torch.unsqueeze(P0[0].cpu(),dim=0).numpy().mean(axis=1)
                test_P1 = torch.unsqueeze(P1[0].cpu(),dim=0).numpy().mean(axis=1)
                test_ua = torch.unsqueeze(ua[0].cpu(),dim=0).numpy().mean(axis=1)
                test_fi = torch.unsqueeze(fai[0].cpu(),dim=0).numpy().mean(axis=1)
                test_output_P1 = torch.unsqueeze(output_p1[0].detach().cpu(),dim=0).numpy().mean(axis=1)
                test_output_ua = torch.unsqueeze(output_ua[0].detach().cpu(),dim=0).numpy().mean(axis=1)
                test_output_fi = torch.unsqueeze(output_fi[0].detach().cpu(),dim=0).numpy().mean(axis=1)

                img_list = [input_P0, test_P1, test_ua, test_fi, test_output_P1, test_output_ua, test_output_fi]

                draw_img(img_list)

                label_and_output_image = np.array(Image.open('test_img.png'))[:, :, 0:3]

                writer.add_image('input_P0', input_P0, len(train_dataloader)*i//50 + train_step // 50, dataformats='CHW')
                writer.add_image('label_P1', test_P1, len(train_dataloader)*i//50 + train_step // 50, dataformats='CHW')
                writer.add_image('output_P1', test_output_P1, len(train_dataloader)*i//50 + train_step // 50, dataformats='CHW')
                writer.add_image('label_and_output', label_and_output_image, len(train_dataloader)*i//50 + train_step // 50, dataformats='HWC')

        # 测试步骤
        ynet.eval()  # 我又没分验证集，怎么测试啊！
        print('--val step--')
        torch.no_grad()  # to increase the validation process uses less memory
        with torch.no_grad():
            ynet.eval()  # 我又没分验证集，怎么测试啊！
            total_val_loss = 0
            for step, [P0, P1, ua, fai] in enumerate(val_dataloader):

                if torch.cuda.is_available():
                    P0 = P0.cuda()
                    P1 = P1.cuda()
                    ua = ua.cuda()
                    fai = fai.cuda()
                output_ua, output_fi, output_p1 = ynet(P0)
                P1 = (ua + 1) * (fai + 1) * 0.5 - 1
                loss_P1 = loss_func_p1(output_p1, P1)
                loss_ua = loss_func_ua(output_ua, ua)
                loss_fi = loss_func_fi(output_fi, fai)
                loss = loss_P1 * 100 + loss_ua * 200 + loss_fi
                total_val_loss += loss.item()
            valid_loss = total_val_loss / len(val_data)
            print("epoch {} val loss {}".format(i, valid_loss))
            writer.add_scalar("val_loss", valid_loss, i)

        # 学习率衰减
        print("learning rate {}".format(lr_scheduler.get_last_lr()[0]))
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()
        # lr_scheduler_end.step()

        if i > 80 or i % 10 == 0:
            torch.save(ynet, "{}/module_{}.pth".format(work_dir, i+1))
            print("saved epoch {}".format(i+1))

    writer.close()

