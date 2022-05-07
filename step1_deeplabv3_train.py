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

def draw_img(img_list, if_show_img=False):
    input_P0, test_P1, test_output = img_list
    plt.figure()
    plt.subplot(1, 3, 1)
    # plt.colorbar()
    plt.title('input_P0', fontsize=12)
    plt.imshow(input_P0.squeeze())
    plt.tick_params(labelsize=5)
    plt.subplot(1, 3, 2)
    plt.title('test_P1', fontsize=12)
    plt.imshow(test_P1.squeeze())
    plt.tick_params(labelsize=5)
    plt.subplot(1, 3, 3)
    plt.title('test_output', fontsize=12)
    plt.imshow(test_output.squeeze())
    plt.tick_params(labelsize=5)
    plt.savefig('latest_img.png')
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

module = deeplabv3_resnet50(aux=False, num_classes=3, pretrain_backbone=False)
print(module)
pretrain = True
if pretrain:
    weights_dict = torch.load("./nets/backbones/deeplabv3_resnet50_coco.pth", map_location='cpu')

    # 官方提供的预训练权重是21类(包括背景)
    # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
    for k in list(weights_dict.keys()):
        if "classifier.4" in k:
            del weights_dict[k]

    missing_keys, unexpected_keys = module.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)


if torch.cuda.is_available():
    # import torch.backends.cudnn as cudnn
    # net = nn.DataParallel(module)  # 我又不用多GPU
    torch.backends.cudnn.benchmark = True
    module = module.cuda()

# 等下我没看懂原文，就是用均方吗？！
loss_mse = nn.MSELoss()

lr = 1e-4
optimizer = optim.Adam(module.parameters(), lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=pow(0.1, 1/10)) # 10次衰减到0.1

work_dir = './workdir/deeplabv3'
writer = SummaryWriter("{}/logs".format(work_dir))

epoch=49
for i in range(epoch):
    print("-------epoch  {} -------".format(i+1))
    # 训练步骤
    module.eval()
    for step, [P0, P1] in enumerate(train_dataloader):
        # print("P0 info {} {} {}".format(P0.size(),P0.max(),P0.min()) )
        # print("P1 info {} {} {}".format(P1.size(),P1.max(),P1.min()) )
        if torch.cuda.is_available():
            P0 = P0.cuda()
            P1 = P1.cuda()
        output_dict = module(P0)
        output = output_dict['out']
        loss = loss_mse(output, P1)
        # print("step {} output tensor {} with loss = {}".format(step,output.size(),loss))
        # 反传优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step = len(train_dataloader)*i+step+1
        if train_step % 50 == 0:
            print("train time：{}, Loss: {}".format(train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)
            # test_P0 = P0.cpu().numpy()[0,:,:,:]
            # test_P1 = P1.cpu().numpy()[0,:,:,:]
            # test_output = output.detach().cpu().numpy()[0,:,:,:]
            # writer.add_image('input_P0', test_P0, train_step/50, dataformats='CHW')
            # writer.add_image('label_P1', test_P1, train_step/50, dataformats='CHW')
            # writer.add_image('output_P1', test_output, train_step/50, dataformats='CHW')
            input_P0 = P0.cpu().numpy().mean(axis=1)
            test_P1 = P1.cpu().numpy().mean(axis=1)
            test_output = output.detach().cpu().numpy().mean(axis=1)

            img_list = [input_P0, test_P1, test_output]

            draw_img(img_list)
            label_and_output_image = np.array(Image.open('latest_img.png'))[:, :, 0:3]
            writer.add_image('label_and_output', label_and_output_image,
                             len(train_dataloader) * i // 50 + train_step / 50, dataformats='HWC')
            # writer.add_image('input_P0', input_P0, len(train_dataloader)*i//50 + train_step/50, dataformats='CHW')
            # writer.add_image('label_P1', test_P1, len(train_dataloader)*i//50 + train_step/50, dataformats='CHW')
            # writer.add_image('output_P1', test_output, len(train_dataloader)*i//50 + train_step/50, dataformats='CHW')
    # 测试步骤
    module.eval()  # 我又没分验证集，怎么测试啊！
    total_test_loss = 0
    total_accuracy = 0

    # 最后学习率衰减
    lr_scheduler.step()

    torch.save(module, "{}/module_{}.pth".format(work_dir,i+1))
    print("saved epoch {}".format(i+1))

writer.close()


