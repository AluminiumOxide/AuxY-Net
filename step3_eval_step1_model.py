from data.Unet_Dataset import Unet_Dataset
from nets.deeplabv3 import deeplabv3_resnet50
from nets.unet_more import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net

from torch.utils.data import DataLoader
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import skimage.metrics

def convert_to_uint8(input_image):
    # drive numpy image from float32 to uint8
    output_image = np.uint8((input_image - input_image.min()) / (input_image.max() - input_image.min()) * 255)
    return output_image

Cuda = True

train_dir_p0 = "./datasets/brain_new/train_p0"
train_dir_ua = "./datasets/brain_new/train_ua"
val_dir_p0 = "./datasets/brain_new/val_p0"
val_dir_ua = "./datasets/brain_new/val_ua"

# train_data = Q_PAT(dir_p0, 'p0', dir_p0_1, 'p0_1')

val_dataset = Unet_Dataset(val_dir_p0, 'p0', val_dir_ua, 'ua_true')
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]
# def model_unet(model_input, in_channel=3, out_channel=1):
#     model_test = model_input(in_channel, out_channel)
#     return model_test
# model_test = model_unet(model_Inputs[0], 3, 3)
#
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True
#     model_test = model_test.cuda()

# load model weights
# assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
# model_test.load_state_dict(torch.load(weights_path))
weights_path_U_Net = "./workdir/U_Net/module_49.pth"
weights_path_R2U_Net = "./workdir/R2U_Net/module_49.pth"
weights_path_AttU_Net = "./workdir/AttU_Net/module_49.pth"
weights_path_R2AttU_Net = "./workdir/R2AttU_Net/module_49.pth"
weights_path_NestedU_Net = "./workdir/NestedUNet/module_49.pth"
weights_path_PSP_Net = "./workdir_step1/pspnet_resnet_df8/module_100.pth"
weights_path_DeepLab_Net = "./workdir_step1/deeplabv3/module_100.pth"
# weights_path_DP_Net = "./workdir/U_Net/module_49.pth"
test_unet = torch.load(weights_path_U_Net)
test_R2U_Net = torch.load(weights_path_R2U_Net)
test_AttU_Net = torch.load(weights_path_AttU_Net)
test_R2AttU_Net = torch.load(weights_path_R2AttU_Net)
test_NestedU_Net = torch.load(weights_path_NestedU_Net)
test_PSP_Net = torch.load(weights_path_PSP_Net)
test_DeepLab_Net = torch.load(weights_path_DeepLab_Net)

model_list = [test_unet, test_R2U_Net, test_AttU_Net, test_R2AttU_Net, test_NestedU_Net, test_PSP_Net ,test_DeepLab_Net]

total_info_list = None
total_columns_name = []
for i, model_category in enumerate(model_list):
    model_name = model_category.__class__.__name__
    total_columns_name.append(model_name+'_ssim')
    total_columns_name.append(model_name+'_psnr')
    total_columns_name.append(model_name+'_mse')
    total_columns_name.append(model_name+'_nr_mse')

    # collect info
    model_info_category = []
    for P0, P1 in valid_loader:
        if torch.cuda.is_available():
            P0 = P0.cuda()
            P1 = P1.cuda()
        output = model_category(P0)

        if model_name == 'DeepLabV3':
            output = output['out']

        input_P0 = P0.cpu().numpy().mean(axis=1)
        test_P1 = P1.cpu().numpy().mean(axis=1)
        test_output = output.cpu().detach().numpy().mean(axis=1)
        input_P0_uint8 = convert_to_uint8(input_P0)
        test_P1_uint8 = convert_to_uint8(test_P1)
        test_output_uint8 = convert_to_uint8(test_output)

        # SSIM值的范围为0至1，越大代表图像越相似。如果两张图片完全一样时，SSIM值为1。
        ssim_score = ssim(test_P1_uint8.squeeze(), test_output_uint8.squeeze(), data_range=255)
        # PSNR的单位为dB，其值越大，图像失真越少。
        psnr_score = psnr(test_P1_uint8.squeeze(), test_output_uint8.squeeze(), data_range=255)
        # 这个就不用多说了
        mse_score = mse(test_P1_uint8.squeeze(), test_output_uint8.squeeze())
        nr_mse_score = skimage.metrics.normalized_root_mse(test_P1_uint8.squeeze(), test_output_uint8.squeeze())
        print('{} info -- ssim: {:.4f} psnr: {:.4f} mse: {:.4f}  normalize_mse: {:.4f}'.format(
            model_name, ssim_score,psnr_score,mse_score,nr_mse_score))
        model_info_category.append([ssim_score, psnr_score, mse_score/255, nr_mse_score])

    # add to main list
    if not total_info_list:  # 若是空的
        total_info_list = model_info_category
    else:  # 假如不是第一次循环
        for index in range(len(model_info_category)):
            total_info_list[index].extend(model_info_category[index])

import pandas as pd
save_file = pd.DataFrame(columns=total_columns_name, data=total_info_list)
print(save_file)
save_file.to_csv('eval_model_info.csv', encoding='gbk')

