
import torch
import torch.nn as nn
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, in_channels=3, pretrained=False):
        super(Unet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], 3, 1)

    def forward(self, inputs):
        feat1 = self.vgg.features[:4](inputs)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final


class DP_end(nn.Module):
    def __init__(self):
        super(DP_end, self).__init__()
        self.conv_end = nn.Conv2d(3, 3, 1)

    def forward(self, ua, fi):
        p1 = (ua + 1) * (fi + 1) * 0.5 - 1
        p1 = self.conv_end(p1)
        return p1


class DP_Net(nn.Module):
    def __init__(self):
        super(DP_Net, self).__init__()
        self.unet_1 = Unet(in_channels=1)
        self.unet_2 = Unet(in_channels=1)
        self.end = DP_end()

    def forward(self, inputs):
        ua = self.unet_1(inputs)
        fi = self.unet_2(inputs)
        p1 = self.end(ua.data, fi.data)
        return p1, ua, fi

    # def optimize_parameters(self):
    #     """Calculate losses, gradients, and update network weights; called in every training iteration"""
    #     self.forward()      # compute fake images and reconstruction images.
    #     self.set_requires_grad(self.unet_1, True)
    #     self.set_requires_grad(self.unet_2, False)
    #     self.set_requires_grad(self.unet_1, False)
    #     self.set_requires_grad(self.unet_2, True)


class DP_Net_new(nn.Module):
    def __init__(self):
        super(DP_Net_new, self).__init__()
        self.unet_1 = Unet(in_channels=3)
        self.unet_2 = Unet(in_channels=3)
        self.end = DP_end()

    def forward(self, inputs):
        ua = self.unet_1(inputs)
        fi = self.unet_2(inputs)
        p1 = self.end(ua.data, fi.data)
        return ua, fi, p1




