import torch
import torch.nn as nn
from nets.vgg import VGG16


class Attention_block(nn.Module):
    def __init__(self, size_gate, size_into, size_middle):
        super(Attention_block, self).__init__()
        self.weight_gate = nn.Sequential(
            nn.Conv2d(size_gate, size_middle, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(size_middle)
        )
        self.weight_into = nn.Sequential(
            nn.Conv2d(size_into, size_middle, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(size_middle)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(size_middle, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, into):
        g1 = self.weight_gate(gate)
        x1 = self.weight_into(into)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = into * psi
        return out


class unet_Up_Attn(nn.Module):
    def __init__(self, in_size_deep, in_size_ceil, out_size):
        super(unet_Up_Attn, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Attn = Attention_block(in_size_deep, in_size_ceil, in_size_ceil//2)
        self.conv1 = nn.Conv2d(in_size_deep + in_size_ceil, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)

    def forward(self, ceil, deep):
        deep_up = self.up(deep)    # deep作为门控gate, ceil作为输入into
        ceil_attn = self.Attn(deep_up, ceil)  # ceil_attn和ceil大小一致
        outputs = torch.cat([ceil_attn, deep_up], 1)  # 通道拼接
        outputs = self.conv1(outputs)  # 因为拼接了,在这里通道调整
        outputs = self.conv2(outputs)  # 加一个特征融合
        return outputs


class DP_end(nn.Module):
    def __init__(self):
        super(DP_end, self).__init__()
        self.conv_end = nn.Conv2d(3, 3, 1)

    def forward(self, ua, fi):
        p1 = (ua + 1) * (fi + 1) * 0.5 - 1
        p1 = self.conv_end(p1)
        return p1


class Ynet(nn.Module):
    def __init__(self, in_channels=3, pretrained=False):
        super(Ynet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_deep = [128, 256, 512, 512]
        in_ceil = [64, 128, 256, 512]
        out_filters = [64, 128, 256, 512]
        # up_sample_ua
        self.ua_up_concat4 = unet_Up_Attn(in_deep[3], in_ceil[3], out_filters[3])  # 64,64,512
        self.ua_up_concat3 = unet_Up_Attn(in_deep[2], in_ceil[2], out_filters[2])  # 128,128,256
        self.ua_up_concat2 = unet_Up_Attn(in_deep[1], in_ceil[1], out_filters[1])  # 256,256,128
        self.ua_up_concat1 = unet_Up_Attn(in_deep[0], in_ceil[0], out_filters[0])  # 512,512,64
        self.ua_final = nn.Conv2d(out_filters[0], 3, 1)  # final conv (without any concat)
        # up_sample_fi
        self.fi_up_concat4 = unet_Up_Attn(in_deep[3], in_ceil[3], out_filters[3])
        self.fi_up_concat3 = unet_Up_Attn(in_deep[2], in_ceil[2], out_filters[2])
        self.fi_up_concat2 = unet_Up_Attn(in_deep[1], in_ceil[1], out_filters[1])
        self.fi_up_concat1 = unet_Up_Attn(in_deep[0], in_ceil[0], out_filters[0])
        self.fi_final = nn.Conv2d(out_filters[0], 3, 1)
        # p1
        self.end = DP_end()

    def forward(self, inputs):  # 假设输入为 512 512 3 ,虽然实际上调整为了256,256,3 宽高需要除2
        feat1 = self.vgg.features[:4](inputs)  # 512,512,64
        feat2 = self.vgg.features[4:9](feat1)  # 256,256,128
        feat3 = self.vgg.features[9:16](feat2)  # 128,128,256
        feat4 = self.vgg.features[16:23](feat3)  # 64,64,512
        feat5 = self.vgg.features[23:-1](feat4)  # 32,32,512

        ua_up4 = self.ua_up_concat4(feat4, feat5)
        ua_up3 = self.ua_up_concat3(feat3, ua_up4)
        ua_up2 = self.ua_up_concat2(feat2, ua_up3)
        ua_up1 = self.ua_up_concat1(feat1, ua_up2)
        ua_final = self.ua_final(ua_up1)

        fi_up4 = self.fi_up_concat4(feat4, feat5)
        fi_up3 = self.fi_up_concat3(feat3, fi_up4)
        fi_up2 = self.fi_up_concat2(feat2, fi_up3)
        fi_up1 = self.fi_up_concat1(feat1, fi_up2)
        fi_final = self.fi_final(fi_up1)

        p1_final = self.end(ua_final.data, fi_final.data)

        return ua_final, fi_final, p1_final



