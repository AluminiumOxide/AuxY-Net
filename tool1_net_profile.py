from nets.unet import Unet
from nets.DP_Net import DP_Net_new
from nets.pspnet import PSPNet
from nets.deeplabv3 import deeplabv3_resnet50
from nets.unet_more import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
import torch

from thop import profile
import pandas as pd

if __name__ == '__main__':

    model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]
    def model_unet(model_input, in_channel=3, out_channel=1):
        model_test = model_input(in_channel, out_channel)
        return model_test
    model_U_Net = model_unet(model_Inputs[0], 3, 3)
    model_R2U_Net = model_unet(model_Inputs[1], 3, 3)
    model_AttU_Net = model_unet(model_Inputs[2], 3, 3)
    model_R2AttU_Net = model_unet(model_Inputs[3], 3, 3)
    model_NestedUNet = model_unet(model_Inputs[4], 3, 3)
    model_dp_net = DP_Net_new()
    model_pspnet = PSPNet(num_classes=3, backbone="resnet50", downsample_factor=8, pretrained=False,
                       aux_branch=False)
    model_deeplab = deeplabv3_resnet50(aux=False, num_classes=3, pretrain_backbone=False)

    model_list = [model_U_Net,model_R2U_Net,model_AttU_Net,model_R2AttU_Net,model_NestedUNet,model_dp_net,model_pspnet, model_deeplab]
    model_info_list = []
    for i,model_category in enumerate(model_list):
        input_test = torch.randn(1, 3, 256, 256)
        macs, params = profile(model_category, (input_test,))
        model_name = model_category.__class__.__name__
        model_info_list.append([model_name, macs, params])

    save_file = pd.DataFrame(columns=['model', 'macs', 'params'], data=model_info_list)
    print(save_file)
    save_file.to_csv('net_profile_csv.csv', encoding='gbk')

    # for info in model_info_list:
    #     print('model:', info[0], 'flops: ', info[1], 'params: ', info[2])

