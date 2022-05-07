from data.Unet_Dataset import Unet_Dataset
from data.DP_Dataset import DP_Dataset
from nets.deeplabv3 import deeplabv3_resnet50
from nets.unet_more import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net

from torch.utils.data import DataLoader
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import skimage.metrics
import os
from scipy.io import savemat


def convert_to_uint8(input_image):
    # drive numpy image from float32 to uint8
    output_image = np.uint8((input_image - input_image.min()) / (input_image.max() - input_image.min()) * 255)
    return output_image

Cuda = True

# brain_new | cylinder_imitation | mouse
dataset_dir = './datasets/cylinder_imitation/'  # dataset root
# brain | cylinder | mouse
data_name = 'cylinder'  # save name

image_path = './eval_result/'+data_name+'/'
if os.path.exists(image_path):
    pass
else:
    os.mkdir(image_path)

train_dir_p0 = dataset_dir + 'train_p0'  # 128_0505
train_dir_fai = dataset_dir + 'train_fai'
train_dir_ua = dataset_dir + 'train_ua'
train_dir_p1 = dataset_dir + 'train_p1'

val_dir_p0 = dataset_dir + 'val_p0'  # 128_0505
val_dir_fai = dataset_dir + 'val_fai'
val_dir_ua = dataset_dir + 'val_ua'
val_dir_p1 = dataset_dir + 'val_p1'

# train_data = Q_PAT(dir_p0, 'p0', dir_p0_1, 'p0_1')

val_data = DP_Dataset(val_dir_p0, 'p0', val_dir_p1, 'p0_1', val_dir_ua, 'ua_true', val_dir_fai, 'fai_1')
val_dataloader = DataLoader(val_data, batch_size=1)

# weights_path_U_Net = "./workdir/U_Net/module_49.pth"

weights_path_U_Net = "./workdir/u_net_"+data_name+"/module_100.pth"
weights_path_DP_Net = "./workdir/dp_net_"+data_name+"/module_100.pth"
weights_path_Y_Net = "./workdir/ynet_"+data_name+"/module_100.pth"
# weights_path_DP_Net = "./workdir/U_Net/module_49.pth"
test_U_Net = torch.load(weights_path_U_Net)
test_DP_Net = torch.load(weights_path_DP_Net)
test_Y_Net = torch.load(weights_path_Y_Net)

# mul_model_list = [test_DP_Net, test_Y_Net]
model_list = [test_U_Net]
total_info_list = None
total_columns_name = []

for i, model_category in enumerate(model_list):
    model_name = model_category.__class__.__name__
    for mul_name in ['ua']:
        total_columns_name.append(model_name+'_ssim_'+mul_name)
        total_columns_name.append(model_name+'_psnr_'+mul_name)
        total_columns_name.append(model_name+'_mse_'+mul_name)
        total_columns_name.append(model_name+'_nrmse_'+mul_name)

    # collect info
    model_info_category = []
    image_index = 0
    for P0, P1, ua, fai in val_dataloader:

        if torch.cuda.is_available():
            P0 = P0.cuda()
            P1 = P1.cuda()
            ua = ua.cuda()
            fai = fai.cuda()
        output_ua = model_category(P0)

        input_p0 = P0.cpu().numpy().mean(axis=1)
        test_ua = ua.cpu().numpy().mean(axis=1)
        test_output_ua = output_ua.cpu().detach().numpy().mean(axis=1)

        input_p0_path = os.path.join(image_path,str(image_index)+'_'+model_name+'_input_p0.mat')
        label_ua_path = os.path.join(image_path,str(image_index)+'_'+model_name+'_label_ua.mat')
        output_ua_path = os.path.join(image_path,str(image_index)+'_'+model_name+'_output_ua.mat')
        savemat(input_p0_path, {'input_p0': np.squeeze(input_p0)})
        savemat(label_ua_path, {'label_ua': np.squeeze(test_ua)})
        savemat(output_ua_path, {'output_ua': np.squeeze(test_output_ua)})
        image_index = image_index + 1

        input_p0_uint8 = convert_to_uint8(input_p0)
        test_ua_uint8 = convert_to_uint8(test_ua)
        test_output_ua_uint8 = convert_to_uint8(test_output_ua)

        test_images = [
                       [test_ua_uint8, test_output_ua_uint8],
                       ]

        line_info = []
        for [label, output] in (test_images):
            ssim_score = ssim(label.squeeze(), output.squeeze(), data_range=255)
            psnr_score = psnr(label.squeeze(), output.squeeze(), data_range=255)
            mse_score = mse(label.squeeze(), output.squeeze())
            nr_mse_score= skimage.metrics.normalized_root_mse(label.squeeze(), output.squeeze())

            print('{} info -- ssim: {:.4f} psnr: {:.4f} mse: {:.4f}  normalize_mse: {:.4f}'.format(
                model_name, ssim_score,psnr_score,mse_score,nr_mse_score))
            line_info.extend([ssim_score, psnr_score, mse_score/255, nr_mse_score])

        model_info_category.append(line_info)

    # add to main list
    if not total_info_list:  # 若是空的
        total_info_list = model_info_category
    else:  # 假如不是第一次循环
        for index in range(len(model_info_category)):
            total_info_list[index].extend(model_info_category[index])


import pandas as pd
save_file = pd.DataFrame(columns=total_columns_name, data=total_info_list)
print(save_file)
save_file.to_csv('./eval_result/eval_mul_model_info_unet_'+data_name+'.csv', encoding='gbk')

