import torchvision
from data.DP_Dataset import DP_Dataset
from nets.unet import Unet
from nets.DP_Net import DP_Net
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np


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


if __name__ == '__main__':

    Cuda = True

    dir_p0 = "../datasets/brain_and_brain/p0"  # 128_0505
    dir_fai = "../datasets/brain_and_brain/fai"
    dir_ua = "../datasets/brain_and_brain/ua"
    dir_p0_1 = "../datasets/brain_and_brain/p0_1"


    # train_data = Q_PAT(dir_p0, 'p0', dir_p0_1, 'p0_1')
    train_data = DP_Dataset(dir_p0, 'p0', dir_p0_1, 'p0_1', dir_ua, 'ua_true', dir_fai, 'fai_1')
    train_dataloader = DataLoader(train_data, batch_size=1)

    unet = Unet(in_channels=3, pretrained=None)
    dp_net = DP_Net()
    if torch.cuda.is_available():
        # import torch.backends.cudnn as cudnn
        # net = nn.DataParallel(module)  # 我又不用多GPU
        torch.backends.cudnn.benchmark = True
        dp_net = dp_net.cuda()

    # 等下我没看懂原文，就是用均方吗？！
    loss_mse = nn.MSELoss()

    lr = 1e-4
    optimizer_unet_1 = optim.Adam(dp_net.unet_1.parameters(), lr)
    optimizer_unet_2 = optim.Adam(dp_net.unet_2.parameters(), lr)
    optimizer_end = optim.Adam(dp_net.end.parameters(), lr)

    lr_scheduler_unet_1 = optim.lr_scheduler.StepLR(optimizer_unet_1, step_size=2, gamma=pow(0.1,1/10)) # 10次衰减到0.1
    lr_scheduler_unet_2 = optim.lr_scheduler.StepLR(optimizer_unet_2, step_size=2, gamma=pow(0.1,1/10)) # 10次衰减到0.1
    lr_scheduler_end = optim.lr_scheduler.StepLR(optimizer_end, step_size=2, gamma=pow(0.1,1/10)) # 10次衰减到0.1

    work_dir = '../workdir'
    writer = SummaryWriter("{}/logs_dp_1".format(work_dir))

    epoch=19
    print("Full dp_net build with unet_1,unet_2,end")
    print("ua  for dp_net.unet_1 \nfai for dp_net.unet_2\nP1  for all dp_net")
    print("Begin train with {} epoch".format(epoch))
    for i in range(epoch):
        print("-------epoch  {} -------".format(i+1))
        # 训练步骤
        dp_net.train()
        for step, [P0, P1, ua, fai] in enumerate(train_dataloader):
            # print("P0 info {} {} {}".format(P0.size(),P0.max(),P0.min()) )
            # print("P1 info {} {} {}".format(P1.size(),P1.max(),P1.min()) )
            if torch.cuda.is_available():
                P0 = P0.cuda()
                P1 = P1.cuda()
                ua = ua.cuda()
                fai = fai.cuda()
            # forward
            output_p1, output_ua, output_fi = dp_net(P0)
            loss_P1 = loss_mse(output_p1, P1)
            loss_ua = loss_mse(output_ua, ua)
            loss_fi = loss_mse(output_fi, fai)
            # 更新ua unet_1
            set_requires_grad([dp_net.unet_2, dp_net.end], False)
            set_requires_grad([dp_net.unet_1], True)
            optimizer_unet_1.zero_grad()
            loss_ua.backward()
            optimizer_unet_1.step()
            # 更新fai unet_2
            set_requires_grad([dp_net.unet_1, dp_net.end], False)
            set_requires_grad([dp_net.unet_2], True)
            optimizer_unet_2.zero_grad()
            loss_fi.backward()
            optimizer_unet_2.step()
            # 更新p1 全部
            set_requires_grad([dp_net], True)
            optimizer_unet_1.zero_grad()
            optimizer_unet_2.zero_grad()
            optimizer_end.zero_grad()
            loss_P1.backward()
            optimizer_end.step()
            optimizer_unet_1.step()
            optimizer_unet_2.step()

            train_step = len(train_dataloader) * i + step + 1

            # self.optimizer_G.zero_grad()
            # self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            # self.backward_G()             # calculate gradients for G_A and G_B
            # self.optimizer_G.step()       # update G_A and G_B's weights
            # # D_A and D_B
            # self.set_requires_grad([self.netD_A, self.netD_B], True)
            # self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            # self.backward_D_A()      # calculate gradients for D_A
            # self.backward_D_B()      # calculate graidents for D_B
            # self.optimizer_D.step()  # update D_A and D_B's weights
            # loss = loss_mse(output, P1)
            # # print("step {} output tensor {} with loss = {}".format(step,output.size(),loss))
            # # 反传优化
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if train_step % 50 == 0:
                print("train time：{}, Loss unet_1|ua:{}, unet_2|fai:{}, all|P1:{}".format(
                    train_step, loss_ua.item(), loss_fi.item(), loss_P1.item()))
                writer.add_scalar("train_loss", loss_ua.item(), train_step)
                writer.add_scalar("train_loss", loss_fi.item(), train_step)
                writer.add_scalar("train_loss", loss_P1.item(), train_step)
                input_P0 = P0.cpu().numpy()[0,:,:,:]
                test_P1 = P1.cpu().numpy()[0,:,:,:]
                test_ua = ua.cpu().numpy()[0,:,:,:]
                test_fi = fai.cpu().numpy()[0,:,:,:]
                test_output_P1 = output_p1.detach().cpu().numpy()[0,:,:,:]
                test_output_ua = output_ua.detach().cpu().numpy()[0,:,:,:]
                test_output_fi = output_fi.detach().cpu().numpy()[0,:,:,:]

                # writer.add_image('input_P0', test_P0, train_step/50, dataformats='CHW')
                # writer.add_image('label_P1', test_P1, train_step/50, dataformats='CHW')
                # writer.add_image('output_P1', test_output, train_step/50, dataformats='CHW')
                writer.add_image('input_P0', input_P0, len(train_dataloader)*i//50 + train_step/50, dataformats='CHW')
                writer.add_image('label_P1', test_P1, len(train_dataloader)*i//50 + train_step/50, dataformats='CHW')
                writer.add_image('output_P1', test_output_P1, len(train_dataloader)*i//50 + train_step/50, dataformats='CHW')
        # 测试步骤
        dp_net.eval()  # 我又没分验证集，怎么测试啊！
        total_test_loss = 0
        total_accuracy = 0

        torch.save(dp_net, "{}/module_{}.pth".format(work_dir,i+1))
        print("saved epoch {}".format(i+1))

    writer.close()



