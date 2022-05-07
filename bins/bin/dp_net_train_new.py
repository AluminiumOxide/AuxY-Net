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


def draw_img(img_list, if_show_img=False):
    input_P0, test_P1, test_ua, test_fi, test_output_P1, test_output_ua, test_output_fi = img_list
    plt.figure()
    plt.subplot(2, 4, 1)
    # plt.colorbar()
    plt.title('label_p1', fontsize=12)
    plt.imshow(test_P1.squeeze())
    plt.tick_params(labelsize=5)
    plt.subplot(2, 4, 2)
    plt.title('label_ua', fontsize=12)
    plt.imshow(test_ua.squeeze())
    plt.tick_params(labelsize=5)
    plt.subplot(2, 4, 3)
    plt.title('label_fi', fontsize=12)
    plt.imshow(test_fi.squeeze())
    plt.tick_params(labelsize=5)
    plt.subplot(2, 4, 5)
    plt.title('output_p1', fontsize=12)
    plt.imshow(test_output_P1.squeeze())
    plt.tick_params(labelsize=5)
    plt.subplot(2, 4, 6)
    plt.title('output_ua', fontsize=12)
    plt.imshow(test_output_ua.squeeze())
    plt.tick_params(labelsize=5)
    plt.subplot(2, 4, 7)
    plt.title('output_fi', fontsize=12)
    plt.imshow(test_output_fi.squeeze())
    plt.tick_params(labelsize=5)
    plt.savefig('latest_img.png')
    plt.subplot(2, 4, 4)
    plt.title('input_p0', fontsize=12)
    plt.imshow(input_P0.squeeze())
    plt.tick_params(labelsize=5)
    plt.savefig('latest_img.png')
    if if_show_img:
        plt.show()
    plt.cla()
    plt.close("all")


if __name__ == '__main__':

    Cuda = True

    dir_p0 = "../datasets/brain_and_brain/p0"  # 128_0505
    dir_fai = "../datasets/brain_and_brain/fai"
    dir_ua = "../datasets/brain_and_brain/ua"
    dir_p0_1 = "../datasets/brain_and_brain/p0_1"


    # train_data = Q_PAT(dir_p0, 'p0', dir_p0_1, 'p0_1')
    train_data = DP_Dataset(dir_p0, 'p0', dir_p0_1, 'p0_1', dir_ua, 'ua_true', dir_fai, 'fai_1')
    train_dataloader = DataLoader(train_data, batch_size=1)

    # unet = Unet(in_channels=3, pretrained=None)
    dp_net = DP_Net_new()
    if torch.cuda.is_available():
        # import torch.backends.cudnn as cudnn
        # net = nn.DataParallel(module)  # 我又不用多GPU
        torch.backends.cudnn.benchmark = True
        dp_net = dp_net.cuda()

    # 等下我没看懂原文，就是用均方吗？！
    loss_func_ua = nn.MSELoss()
    loss_func_fi = nn.MSELoss()
    loss_func_p1 = nn.MSELoss()

    lr = 1e-5
    optimizer_unet_1 = optim.Adam(dp_net.unet_1.parameters(), lr)
    optimizer_unet_2 = optim.Adam(dp_net.unet_2.parameters(), lr)
    optimizer = optim.Adam(dp_net.parameters(), lr)

    lr_scheduler_unet_1 = optim.lr_scheduler.StepLR(optimizer_unet_1, step_size=3, gamma=pow(0.1, 1/10))  # 10次衰减到0.1
    lr_scheduler_unet_2 = optim.lr_scheduler.StepLR(optimizer_unet_2, step_size=3, gamma=pow(0.1, 1/10))  # 10次衰减到0.1
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=pow(0.1, 1/10))  # 10次衰减到0.1

    work_dir = './workdir/dp_net'
    writer = SummaryWriter("{}/logs".format(work_dir))

    epoch = 29
    print("Full dp_net build with unet_1,unet_2,end")
    print("ua  for dp_net.unet_1 \nfai for dp_net.unet_2\nP1  for all dp_net")
    print("Begin train with {} epoch".format(epoch))
    for i in range(epoch):
        print("-------epoch  {} -------".format(i+1))
        # 训练步骤
        dp_net.train()
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
            output_ua, output_fi = dp_net(P0)
            # output_p1 = output_ua.data * output_fi.data
            output_p1 = output_ua * output_fi
            P1 = ua.data *fai.data
            # loss_P1 = loss_func_p1(output_p1, P1)
            # loss_ua = loss_func_ua(output_ua, ua)
            # loss_fi = loss_func_fi(output_fi, fai)
            # loss_P1.requires_grad_(True)
            # loss_ua.requires_grad_(True)
            # loss_fi.requires_grad_(True)

            # 更新ua unet_1
            set_requires_grad([dp_net.unet_2], False)
            set_requires_grad([dp_net.unet_1], True)
            optimizer_unet_1.zero_grad()
            loss_ua = loss_func_ua(output_ua, ua)
            loss_ua.requires_grad_(True)
            loss_ua.backward(retain_graph=True)  # 前面加了,这里似乎就不需要了 loss_ua.requires_grad = True
            optimizer_unet_1.step()

            # 更新fai unet_2

            set_requires_grad([dp_net.unet_1], False)
            set_requires_grad([dp_net.unet_2], True)
            optimizer_unet_2.zero_grad()
            loss_fi = loss_func_fi(output_fi, fai)
            loss_fi.requires_grad_(True)
            loss_fi.backward(retain_graph=True)
            optimizer_unet_2.step()

            # 更新p1 (感觉不用更新啊)
            set_requires_grad([dp_net.unet_1, dp_net.unet_2], True)
            # set_requires_grad([dp_net.end], True)
            optimizer.zero_grad()
            loss_P1 = loss_func_p1(output_p1, P1)
            loss_P1.requires_grad_(True)
            loss_ua.requires_grad_(True)
            loss_fi.requires_grad_(True)
            loss_all = loss_ua + loss_fi + loss_P1
            loss_all.backward(retain_graph=True)
            optimizer_unet_2.step()

            # optimizer_unet_1.zero_grad()
            # optimizer_unet_2.zero_grad()
            # optimizer_end.zero_grad()

            # loss_P1.backward()
            # optimizer_end.step()
            # optimizer_unet_1.step()
            # optimizer_unet_2.step()

            train_step = len(train_dataloader) * i + step + 1

            if train_step % 50 == 0:
                print("train time：{}, Loss unet_1|ua:{:.4f}, unet_2|fai:{:.4f}, all|P1:{:.4f}".format(
                    train_step, loss_ua.item(), loss_fi.item(), loss_P1.item()))

                writer.add_scalar("train_loss/ua", loss_ua.item(), train_step)
                writer.add_scalar("train_loss/fi", loss_fi.item(), train_step)
                writer.add_scalar("train_loss/p1", loss_P1.item(), train_step)

                # input_P0 = P0.cpu().numpy()[0,:,:,:]   # P0.cpu().numpy().mean(axis=1)
                # test_P1 = P1.cpu().numpy()[0,:,:,:]
                # test_ua = ua.cpu().numpy()[0,:,:,:]
                # test_fi = fai.cpu().numpy()[0,:,:,:]
                # test_output_P1 = output_p1.detach().cpu().numpy()[0,:,:,:]
                # test_output_ua = output_ua.detach().cpu().numpy()[0,:,:,:]
                # test_output_fi = output_fi.detach().cpu().numpy()[0,:,:,:]
                input_P0 = P0.cpu().numpy().mean(axis=1)
                test_P1 = P1.cpu().numpy().mean(axis=1)
                test_ua = ua.cpu().numpy().mean(axis=1)
                test_fi = fai.cpu().numpy().mean(axis=1)
                test_output_P1 = output_p1.detach().cpu().numpy().mean(axis=1)
                test_output_ua = output_ua.detach().cpu().numpy().mean(axis=1)
                test_output_fi = output_fi.detach().cpu().numpy().mean(axis=1)

                img_list = [input_P0, test_P1, test_ua, test_fi, test_output_P1, test_output_ua, test_output_fi]

                draw_img(img_list)

                label_and_output_image = np.array(Image.open('../latest_img.png'))[:, :, 0:3]

                writer.add_image('input_P0', input_P0, len(train_dataloader)*i//50 + train_step/50, dataformats='CHW')
                writer.add_image('label_P1', test_P1, len(train_dataloader)*i//50 + train_step/50, dataformats='CHW')
                writer.add_image('output_P1', test_output_P1, len(train_dataloader)*i//50 + train_step/50, dataformats='CHW')
                writer.add_image('label_and_output', label_and_output_image, len(train_dataloader)*i//50 + train_step/50, dataformats='HWC')

        # 测试步骤
        dp_net.eval()  # 我又没分验证集，怎么测试啊！
        total_test_loss = 0
        total_accuracy = 0

        # 学习率衰减
        print("learning rate {} {}".format(
            lr_scheduler_unet_1.get_last_lr()[0], lr_scheduler_unet_2.get_last_lr()[0]
        ))
        lr_scheduler_unet_1.step()
        lr_scheduler_unet_2.step()
        lr_scheduler.step()
        # lr_scheduler_end.step()

        torch.save(dp_net, "{}/module_{}.pth".format(work_dir,i+1))
        print("saved epoch {}".format(i+1))

    writer.close()




