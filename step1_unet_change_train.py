import torchvision
from data.Unet_Dataset import Unet_Dataset
from nets.unet_more import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from torch.utils.data.sampler import SubsetRandomSampler
#
# import torchsummary
# #from torch.utils.tensorboard import SummaryWriter
# #from tensorboardX import SummaryWriter


def draw_img(img_list, if_show_img=False):
    input_P0, label_P1, test_output = img_list
    plt.figure(figsize=(8, 3), dpi=240)
    plt.subplot(1, 3, 1)
    # plt.colorbar()
    plt.title('input_P0', fontsize=12)
    plt.imshow(input_P0.squeeze())
    plt.tick_params(labelsize=5)
    plt.subplot(1, 3, 2)
    plt.title('label_ua', fontsize=12)
    plt.imshow(label_P1.squeeze())
    plt.tick_params(labelsize=5)
    plt.subplot(1, 3, 3)
    plt.title('output_ua', fontsize=12)
    plt.imshow(test_output.squeeze())
    plt.tick_params(labelsize=5)
    plt.savefig('latest_img.png')
    if if_show_img:
        plt.show()
    plt.cla()
    plt.close("all")


train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')
device = torch.device("cuda:0" if train_on_gpu else "cpu")

train_dir_p0 = "./datasets/brain_new/train_p0"
train_dir_ua = "./datasets/brain_new/train_ua"
val_dir_p0 = "./datasets/brain_new/val_p0"
val_dir_ua = "./datasets/brain_new/val_ua"
# 创建数据集
train_dataset = Unet_Dataset(train_dir_p0, 'p0', train_dir_ua, 'ua_true')
val_dataset = Unet_Dataset(val_dir_p0, 'p0', val_dir_ua, 'ua_true')
print("{} mat images for train and {} mat images for val".format(len(train_dataset),len(val_dataset)))
# 存入dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# 建立模型
model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]
def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test
#passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary
model_test = model_unet(model_Inputs[0], 3, 3)
model_test.to(device)
# 损失函数和学习率等
loss_mse = nn.MSELoss()
lr = 1e-4
optimizer = optim.Adam(model_test.parameters(), lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=pow(0.1, 1/10)) # 10次衰减到0.1

work_dir = './workdir'
writer = SummaryWriter("{}/logs".format(work_dir))

epoch=49
for i in range(epoch):
    print("-------epoch  {} -------".format(i+1))

    # ----------------------------------------------
    # 训练步骤
    print('--train step--')
    model_test.train()
    for step, [P0, P1] in enumerate(train_loader):
        # print("P0 info {} {} {}".format(P0.size(),P0.max(),P0.min()) )
        # print("P1 info {} {} {}".format(P1.size(),P1.max(),P1.min()) )
        if torch.cuda.is_available():
            P0 = P0.cuda()
            P1 = P1.cuda()
        output = model_test(P0)
        loss = loss_mse(output, P1)
        # print("step {} output tensor {} with loss = {}".format(step,output.size(),loss))
        # 反传优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step = len(train_loader)*i+step+1
        if train_step % 50 == 0:
            print("train time：{}, train Loss: {}".format(train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)
            input_P0 = P0.cpu().numpy().mean(axis=1)
            label_P1 = P1.cpu().numpy().mean(axis=1)
            test_output = output.detach().cpu().numpy().mean(axis=1)

            img_list = [input_P0, label_P1, test_output]
            draw_img(img_list)
            label_and_output_image = np.array(Image.open('latest_img.png'))[:, :, 0:3]

            writer.add_image('label_and_output', label_and_output_image,
                             len(train_loader) * i // 50 + train_step / 50, dataformats='HWC')

    # ----------------------------------------------
    # 测试步骤
    print('--val step--')
    torch.no_grad()  # to increase the validation process uses less memory
    with torch.no_grad():
        model_test.eval()  # 我又没分验证集，怎么测试啊！
        total_val_loss = 0
        for step, [P0, P1] in enumerate(valid_loader):

            if torch.cuda.is_available():
                P0 = P0.cuda()
                P1 = P1.cuda()
            output = model_test(P0)
            loss = loss_mse(output, P1)
            total_val_loss += loss.item() * P0.size(0)
        valid_loss = total_val_loss / len(val_dataset)
        print("epoch {} val loss {}".format(i,valid_loss))
        writer.add_scalar("val_loss", valid_loss, i)

    # 最后学习率衰减
    lr_scheduler.step()
    lr = lr_scheduler.get_last_lr()

    # 保存模型
    torch.save(model_test, "{}/module_{}.pth".format(work_dir,i+1))
    print("saved epoch {}".format(i+1))

writer.close()



