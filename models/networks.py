import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from .sincnet import SincConv
import math

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d, semanticoutput=False):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        if semanticoutput:
            return nn.Sequential(*[upconv])
        else:
            return nn.Sequential(*[upconv, nn.Sigmoid()])

        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)


def create_1dconv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv1d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm1d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class AudioVisualPyramidAttentionAudioDepth(nn.Module):
    """
    Me and Deen's idea on self-attention at each stage of the audio and vision convolution steps
    """
    def __init__(self, audio_conv1x1_dim,
                       audio_shape,
                       audio_feature_length,
                       visual_ngf=64,
                       visual_input_nc=3,
                       output_nc=1):
        super(AudioVisualPyramidAttentionAudioDepth, self).__init__()
        # AUDIO SETUP

        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(6, 4), (4, 4), (4, 4), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(3, 2), (2, 2), (2, 2), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.conv1 = create_conv(self._n_input_audio, 64, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(64, 128, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(128, 256, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        self.conv4 = create_conv(256, 512, kernel=self._cnn_layers_kernel_size[3], paddings=0, stride=self._cnn_layers_stride[3])
        self.conv5 = create_conv(512, 512, kernel=self._cnn_layers_kernel_size[4], paddings=0, stride=self._cnn_layers_stride[4])
        # layers = [self.conv1, self.conv2, self.conv3]
        # self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(1152, audio_feature_length, 1, 0)

        # VISION SETUP

        self.rgbdepth_convlayer1 = unet_conv(visual_input_nc, visual_ngf)
        self.rgbdepth_convlayer2 = unet_conv(visual_ngf, visual_ngf * 2)
        self.rgbdepth_convlayer3 = unet_conv(visual_ngf * 2, visual_ngf * 4)
        self.rgbdepth_convlayer4 = unet_conv(visual_ngf * 4, visual_ngf * 8)
        self.rgbdepth_convlayer5 = unet_conv(visual_ngf * 8, visual_ngf * 8)
        self.rgbdepth_upconvlayer1 = unet_upconv(512, visual_ngf * 8)
        self.rgbdepth_upconvlayer2 = unet_upconv(visual_ngf * 16, visual_ngf *4)
        self.rgbdepth_upconvlayer3 = unet_upconv(visual_ngf * 8, visual_ngf * 2)
        self.rgbdepth_upconvlayer4 = unet_upconv(visual_ngf * 4, visual_ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv(visual_ngf * 2, output_nc, True)


        self.attlayer1 = Self_Attn_Downsample(64, 64, keyprojin=1722, keyprojout=1024, channelnum=1)
        self.attlayer2 = Self_Attn(128, 128, keyprojin=1640, keyprojout=1024, channelnum=1)
        self.attlayer3 = Self_Attn(256, 256, keyprojin=361, keyprojout=256, channelnum=1)
        self.attlayer4 = Self_Attn(512, 512, keyprojin=64, keyprojout=64, channelnum=1)

        self.vtoa_attlayer1 = Self_Attn_Downsample(64, 64, keyprojin=1024, keyprojout=1722, channelnum=1)
        self.vtoa_attlayer2 = Self_Attn(128, 128, keyprojin=1024, keyprojout=1640, channelnum=1)
        self.vtoa_attlayer3 = Self_Attn(256, 256, keyprojin=256, keyprojout=361, channelnum=1)
        self.vtoa_attlayer4 = Self_Attn(512, 512, keyprojin=64, keyprojout=64, channelnum=1)

        self.audiodepth_upconvlayer1 = unet_upconv(512, 512) #1016 (audio-visual feature) = 512 (visual feature) + 504 (audio feature)
        self.audiodepth_upconvlayer2 = unet_upconv(1024, 256)
        self.audiodepth_upconvlayer3 = unet_upconv(512, 128)
        self.audiodepth_upconvlayer4 = unet_upconv(256, 64)
        self.audiodepth_upconvlayer5 = unet_upconv(128, output_nc, True)

        self.testup = unet_upconv(512, 512)

        self.audioavg1 = nn.AdaptiveAvgPool2d((64,64))
        self.audioavg2 = nn.AdaptiveAvgPool2d((32,32))
        self.audioavg3 = nn.AdaptiveAvgPool2d((16,16))
        self.audioavg5 = nn.AdaptiveAvgPool2d((4,4))

        self.vgamma1 = nn.Parameter(torch.zeros(1))
        self.vgamma2 = nn.Parameter(torch.zeros(1))
        self.vgamma3 = nn.Parameter(torch.zeros(1))
        self.vgamma4 = nn.Parameter(torch.zeros(1))

        self.agamma1 = nn.Parameter(torch.zeros(1))
        self.agamma2 = nn.Parameter(torch.zeros(1))
        self.agamma3 = nn.Parameter(torch.zeros(1))
        self.agamma4 = nn.Parameter(torch.zeros(1))
    
    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x_audio, x_rgb):
        # BLOCK 1
        x_audio1 = self.conv1(x_audio)
        # print("audio 1", x_audio1.shape)
        rgb_depth_conv1feature = self.rgbdepth_convlayer1(x_rgb)
        # print("rgb 1", rgb_depth_conv1feature.shape)

        attend1, _ = self.attlayer1(rgb_depth_conv1feature, x_audio1)
        vtoa_attend1, _ = self.vtoa_attlayer1(x_audio1, rgb_depth_conv1feature)

        x_audio1 = x_audio1 + self.agamma1 * vtoa_attend1
        rgb_depth_conv1feature = rgb_depth_conv1feature + self.vgamma1 * attend1
        
        # BLOCK 2

        x_audio2 = self.conv2(x_audio1)
        # print("audio 2", x_audio2.shape)
        rgb_depth_conv2feature = self.rgbdepth_convlayer2(rgb_depth_conv1feature)
        # print("rgb 2", rgb_depth_conv2feature.shape)

        attend2, _ = self.attlayer2(rgb_depth_conv2feature, x_audio2)
        vtoa_attend2, _ = self.vtoa_attlayer2(x_audio2, rgb_depth_conv2feature)

        rgb_depth_conv2feature = rgb_depth_conv2feature + self.vgamma2 * attend2
        x_audio2 = x_audio2 + self.agamma2 * vtoa_attend2

        # BLOCK 3

        x_audio3 = self.conv3(x_audio2)
        # print("audio 3", x_audio3.shape)
        rgb_depth_conv3feature = self.rgbdepth_convlayer3(rgb_depth_conv2feature)
        # print("rgb 3", rgb_depth_conv3feature.shape)

        attend3, _ = self.attlayer3(rgb_depth_conv3feature, x_audio3)
        vtoa_attend3, _ = self.vtoa_attlayer3(x_audio3, rgb_depth_conv3feature)

        rgb_depth_conv3feature = rgb_depth_conv3feature + self.vgamma3 * attend3
        x_audio3 = x_audio3 + self.agamma3 * vtoa_attend3

        # BLOCK 4

        x_audio4 = self.conv4(x_audio3)
        # print("audio 4", x_audio4.shape)
        rgb_depth_conv4feature = self.rgbdepth_convlayer4(rgb_depth_conv3feature)
        # print("rgb 4", rgb_depth_conv4feature.shape)


        # print("===============================================")

        attend4, _ = self.attlayer4(rgb_depth_conv4feature, x_audio4)
        vtoa_attend4, _ = self.vtoa_attlayer4(x_audio4, rgb_depth_conv4feature)

        rgb_depth_conv4feature = rgb_depth_conv4feature + self.vgamma4 * attend4
        x_audio4 = x_audio4 + self.agamma4 * vtoa_attend4

        # BLOCK 5

        x_audio5 = self.conv5(x_audio4)
        # print("audio 5", x_audio5.shape)
        rgb_depth_conv5feature = self.rgbdepth_convlayer5(rgb_depth_conv4feature)
        # print("rgb 5", rgb_depth_conv5feature.shape)
        
        x_audio_avg = self.audioavg5(x_audio5)
        # print("audio averaged", x_audio_avg.shape)

        # audio_feat_flat = x_audio5.view(x_audio5.shape[0], -1, 1, 1)
        # audio_feat_flat = self.conv1x1(audio_feat_flat)
        
        # audio_feat_repeated = audio_feat_flat.repeat(1, 1, rgb_depth_conv5feature.shape[-2], rgb_depth_conv5feature.shape[-1])
        # audioVisual_feature = rgb_depth_conv5feature + audio_feat_repeated

        # audioVisual_feature = x_audio_avg + rgb_depth_conv5feature
        audioVisual_feature = torch.cat((x_audio_avg, rgb_depth_conv5feature), dim=1)


        # print(rgb_depth_conv5feature.shape)
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(rgb_depth_conv5feature)
        # print(rgbdepth_upconv1feature.shape)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(torch.cat((rgbdepth_upconv1feature, rgb_depth_conv4feature), dim=1))
        # print(rgbdepth_upconv2feature.shape)
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(torch.cat((rgbdepth_upconv2feature, rgb_depth_conv3feature), dim=1))
        # print(rgbdepth_upconv3feature.shape)
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(torch.cat((rgbdepth_upconv3feature, rgb_depth_conv2feature), dim=1))
        # print(rgbdepth_upconv4feature.shape)
        vision_depth_prediction = self.rgbdepth_upconvlayer5(torch.cat((rgbdepth_upconv4feature, rgb_depth_conv1feature), dim=1))
        # print(vision_depth_prediction.shape)

        # Audio Depth
        # TODO: TRY ADDING ATTENDED AUDIO HERE
        audiodepth1 = self.audiodepth_upconvlayer1(x_audio_avg)
        # print("first upconf", audiodepth1.shape, x_audio4.shape)
        audiodepth2 = self.audiodepth_upconvlayer2(torch.cat((audiodepth1, x_audio4), dim=1))
        audiodepth3 = self.audiodepth_upconvlayer3(torch.cat((audiodepth2, self.audioavg3(x_audio3)), dim=1))
        audiodepth4 = self.audiodepth_upconvlayer4(torch.cat((audiodepth3, self.audioavg2(x_audio2)), dim=1))
        audio_depth_prediction = self.audiodepth_upconvlayer5(torch.cat((audiodepth4, self.audioavg1(x_audio1)), dim=1))
        # print("final audio", audio_depth_prediction.shape)


        # return vision_depth_prediction, audio_depth_prediction
        return vision_depth_prediction, audio_depth_prediction, audioVisual_feature


class AudioVisualPyramidAttention(nn.Module):
    """
    Me and Deen's idea on self-attention at each stage of the audio and vision convolution steps
    """
    def __init__(self, audio_conv1x1_dim,
                       audio_shape,
                       audio_feature_length,
                       visual_ngf=64,
                       visual_input_nc=3,
                       output_nc=1,
                       vtoa=False):
        super(AudioVisualPyramidAttention, self).__init__()
        # AUDIO SETUP
        self.vtoa = vtoa
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(6, 6), (4, 4), (4, 4), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(3, 3), (2, 2), (2, 2), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, 128, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        self.conv4 = create_conv(128, 256, kernel=self._cnn_layers_kernel_size[3], paddings=0, stride=self._cnn_layers_stride[3])
        self.conv5 = create_conv(256, audio_conv1x1_dim, kernel=self._cnn_layers_kernel_size[4], paddings=0, stride=self._cnn_layers_stride[4])
        # layers = [self.conv1, self.conv2, self.conv3]
        # self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(1152, audio_feature_length, 1, 0)

        # VISION SETUP

        self.rgbdepth_convlayer1 = unet_conv(visual_input_nc, visual_ngf)
        self.rgbdepth_convlayer2 = unet_conv(visual_ngf, visual_ngf * 2)
        self.rgbdepth_convlayer3 = unet_conv(visual_ngf * 2, visual_ngf * 4)
        self.rgbdepth_convlayer4 = unet_conv(visual_ngf * 4, visual_ngf * 8)
        self.rgbdepth_convlayer5 = unet_conv(visual_ngf * 8, visual_ngf * 8)
        self.rgbdepth_upconvlayer1 = unet_upconv(512, visual_ngf * 8)
        self.rgbdepth_upconvlayer2 = unet_upconv(visual_ngf * 16, visual_ngf *4)
        self.rgbdepth_upconvlayer3 = unet_upconv(visual_ngf * 8, visual_ngf * 2)
        self.rgbdepth_upconvlayer4 = unet_upconv(visual_ngf * 4, visual_ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv(visual_ngf * 2, output_nc, True)


        self.attlayer1 = Self_Attn_Downsample(64, 32, keyprojin=42*27, keyprojout=1024, channelnum=4)
        self.attlayer2 = Self_Attn(128, 64, keyprojin=1066, keyprojout=1024, channelnum=16)
        self.attlayer3 = Self_Attn(256, 128, keyprojin=228, keyprojout=256, channelnum=32)
        self.attlayer4 = Self_Attn(512, 256, keyprojin=40, keyprojout=64, channelnum=64)
        # self.attlayer5 = Self_Attn(512, 64, keyprojin=18, keyprojout=16)

        if self.vtoa:
            self.vtoa_attlayer1 = Self_Attn_Downsample(32, 64, keyprojin=1024, keyprojout=1134, channelnum=4)
            self.vtoa_attlayer2 = Self_Attn(64, 128, keyprojin=1024, keyprojout=1066, channelnum=16)
            self.vtoa_attlayer3 = Self_Attn(128, 256, keyprojin=256, keyprojout=228, channelnum=32)
            self.vtoa_attlayer4 = Self_Attn(256, 512, keyprojin=64, keyprojout=40, channelnum=64)
    
    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x_audio, x_rgb):
        # BLOCK 1
        x_audio1 = self.conv1(x_audio)
        rgbdepth_conv1feature = self.rgbdepth_convlayer1(x_rgb)

        attend1, _ = self.attlayer1(rgbdepth_conv1feature, x_audio1)
        if self.vtoa:
            vtoa_attend1, _ = self.vtoa_attlayer1(x_audio1, rgbdepth_conv1feature)

        rgb_depth_conv1feature = rgbdepth_conv1feature + attend1
        if self.vtoa:
            x_audio1 = x_audio1 + vtoa_attend1

        # BLOCK 2

        x_audio2 = self.conv2(x_audio1)
        rgbdepth_conv2feature = self.rgbdepth_convlayer2(rgb_depth_conv1feature)

        attend2, _ = self.attlayer2(rgbdepth_conv2feature, x_audio2)
        if self.vtoa:
            vtoa_attend2, _ = self.vtoa_attlayer2(x_audio2, rgbdepth_conv2feature)

        rgb_depth_conv2feature = rgbdepth_conv2feature + attend2
        if self.vtoa:
            x_audio2 = x_audio2 + vtoa_attend2

        # BLOCK 3

        x_audio3 = self.conv3(x_audio2)
        rgbdepth_conv3feature = self.rgbdepth_convlayer3(rgb_depth_conv2feature)

        attend3, _ = self.attlayer3(rgbdepth_conv3feature, x_audio3)
        if self.vtoa:
            vtoa_attend3, _ = self.vtoa_attlayer3(x_audio3, rgbdepth_conv3feature)

        rgb_depth_conv3feature = rgbdepth_conv3feature + attend3
        if self.vtoa:
            x_audio3 = x_audio3 + vtoa_attend3

        # BLOCK 4

        x_audio4 = self.conv4(x_audio3)
        rgbdepth_conv4feature = self.rgbdepth_convlayer4(rgb_depth_conv3feature)

        attend4, _ = self.attlayer4(rgbdepth_conv4feature, x_audio4)
        if self.vtoa:
            vtoa_attend4, _ = self.vtoa_attlayer4(x_audio4, rgbdepth_conv4feature)

        rgb_depth_conv4feature = rgbdepth_conv4feature + attend4
        if self.vtoa:
            x_audio4 = x_audio4 + vtoa_attend4

        # BLOCK 5

        x_audio5 = self.conv5(x_audio4)
        rgb_depth_conv5feature = self.rgbdepth_convlayer5(rgb_depth_conv4feature)

        # attend5, _ = self.attlayer5(rgb_depth_conv5feature, x_audio5)
        # rgb_depth_conv5feature = rgb_depth_conv5feature + attend5

        audio_feat_flat = x_audio5.view(x_audio5.shape[0], -1, 1, 1)

        audio_feat_flat = self.conv1x1(audio_feat_flat)
        audio_feat_repeated = audio_feat_flat.repeat(1, 1, rgb_depth_conv5feature.shape[-2], rgb_depth_conv5feature.shape[-1])

        audioVisual_feature = rgb_depth_conv5feature + audio_feat_repeated

        # UPCONV BLOCK

        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(audioVisual_feature)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(torch.cat((rgbdepth_upconv1feature, rgbdepth_conv4feature), dim=1))
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(torch.cat((rgbdepth_upconv2feature, rgbdepth_conv3feature), dim=1))
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(torch.cat((rgbdepth_upconv3feature, rgbdepth_conv2feature), dim=1))
        depth_prediction = self.rgbdepth_upconvlayer5(torch.cat((rgbdepth_upconv4feature, rgbdepth_conv1feature), dim=1))

        return depth_prediction

class SimpleAudioDepthNet(nn.Module):
    ## strucure adapted from VisualEchoes [ECCV 2020]
    r"""A Simple 3-Conv CNN followed by a fully connected layer
    """
    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, output_nc=1):
        super(SimpleAudioDepthNet, self).__init__()
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, conv1x1_dim, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        layers = [self.conv1, self.conv2, self.conv3]
        self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(conv1x1_dim * cnn_dims[0] * cnn_dims[1], audio_feature_length, 1, 0)

        self.rgbdepth_upconvlayer1 = unet_upconv(512, 512) #1016 (audio-visual feature) = 512 (visual feature) + 504 (audio feature)
        self.rgbdepth_upconvlayer2 = unet_upconv(512, 256)
        self.rgbdepth_upconvlayer3 = unet_upconv(256, 128)
        self.rgbdepth_upconvlayer4 = unet_upconv(128, 64)
        self.rgbdepth_upconvlayer5 = unet_upconv(64, 32)
        self.rgbdepth_upconvlayer6 = unet_upconv(32, 16)
        self.rgbdepth_upconvlayer7 = unet_upconv(16, output_nc, True)

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x):
        # print("audio shape before feature", x.shape)
        x = self.feature_extraction(x)
        # print("audio shape after feature", x.shape)
        x = x.view(x.shape[0], -1, 1, 1)
        # print("audio shape after view", x.shape)
        x = self.conv1x1(x)
        # print("audio shape 1x1", x.shape)
        
        audio_feat = x
        
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(audio_feat)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(rgbdepth_upconv1feature)
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(rgbdepth_upconv2feature)
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(rgbdepth_upconv3feature)
        rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(rgbdepth_upconv4feature)
        rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature)
        depth_prediction = self.rgbdepth_upconvlayer7(rgbdepth_upconv6feature)
        return depth_prediction, audio_feat


class Simple5LayerAudioDepthNet(nn.Module):
    ## strucure adapted from VisualEchoes [ECCV 2020]
    r"""A Simple 3-Conv CNN followed by a fully connected layer
    """
    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, output_nc=1):
        super(Simple5LayerAudioDepthNet, self).__init__()
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(4, 4), (4, 4), (4, 4), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(2, 2), (2, 2), (2, 2), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, 128, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        self.conv4 = create_conv(128, 256, kernel=self._cnn_layers_kernel_size[3], paddings=0, stride=self._cnn_layers_stride[3])
        self.conv5 = create_conv(256, conv1x1_dim, kernel=self._cnn_layers_kernel_size[4], paddings=0, stride=self._cnn_layers_stride[4])
        # layers = [self.conv1, self.conv2, self.conv3]
        # self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(conv1x1_dim * cnn_dims[0] * cnn_dims[1], audio_feature_length, 1, 0)

        self.rgbdepth_upconvlayer1 = unet_upconv(512, 512) #1016 (audio-visual feature) = 512 (visual feature) + 504 (audio feature)
        self.rgbdepth_upconvlayer2 = unet_upconv(512, 256)
        self.rgbdepth_upconvlayer3 = unet_upconv(256, 128)
        self.rgbdepth_upconvlayer4 = unet_upconv(128, 64)
        self.rgbdepth_upconvlayer5 = unet_upconv(64, 32)
        self.rgbdepth_upconvlayer6 = unet_upconv(32, 16)
        self.rgbdepth_upconvlayer7 = unet_upconv(16, output_nc, True)

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x):
        # print("audio shape", x.shape)
        x = self.conv1(x)
        # x = self.feature_extraction(x)
        # print("audio conv 1", x.shape)
        x = self.conv2(x)
        # print("audio conv 2", x.shape)
        x = self.conv3(x)
        # print("audio conv 3", x.shape)
        x = self.conv4(x)
        # print("audio conv 4", x.shape)
        x = self.conv5(x)
        # print("audio conv 5", x.shape)
        x = x.view(x.shape[0], -1, 1, 1)
        # print("view", x.shape)
        x = self.conv1x1(x)
        # print("1x1conv", x.shape)
        
        audio_feat = x
        
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(audio_feat)
        # print("upconv", rgbdepth_upconv1feature.shape)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(rgbdepth_upconv1feature)
        # print("upconv", rgbdepth_upconv2feature.shape)
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(rgbdepth_upconv2feature)
        # print("upconv", rgbdepth_upconv3feature.shape)
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(rgbdepth_upconv3feature)
        # print("upconv", rgbdepth_upconv4feature.shape)
        rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(rgbdepth_upconv4feature)
        # print("upconv", rgbdepth_upconv5feature.shape)
        rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature)
        # print("upconv", rgbdepth_upconv6feature.shape)
        depth_prediction = self.rgbdepth_upconvlayer7(rgbdepth_upconv6feature)
        # print("upconv", depth_prediction.shape)
        return depth_prediction


class WaveformAudioDepthNet(nn.Module):
    r"""MODIFIED TO BE A WAVEFORM MODEL FOR COMPARISON: 
    
        A Simple 3-Conv CNN followed by a fully connected layer
    """
    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, output_nc=1, use_sincnet=False):
        super(WaveformAudioDepthNet, self).__init__()
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(9), (5), (3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(4), (2), (1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        # for kernel_size, stride in zip(
        #     self._cnn_layers_kernel_size, self._cnn_layers_stride
        # ):
        #     cnn_dims = self._conv_output_dim(
        #         dimension=cnn_dims,
        #         padding=np.array([0, 0], dtype=np.float32),
        #         dilation=np.array([1, 1], dtype=np.float32),
        #         kernel_size=np.array(kernel_size, dtype=np.float32),
        #         stride=np.array(stride, dtype=np.float32),
        #     )
        if use_sincnet:
            self.conv1 = SincConv(out_channels=32, kernel_size=381, in_channels=2, padding="valid", stride=self._cnn_layers_stride[0], sample_rate=44100)
        else:
            self.conv1 = create_1dconv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_1dconv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_1dconv(64, conv1x1_dim, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        layers = [self.conv1, self.conv2, self.conv3]
        self.feature_extraction = nn.Sequential(*layers)

        # print(torch.zeros(audio_shape).unsqueeze(0).shape)
        test_audoutputshape = self.feature_extraction(torch.zeros(audio_shape).unsqueeze(0)).size()
        # print(test_audoutputshape)
        self.conv1x1 = create_conv(test_audoutputshape[1] * test_audoutputshape[2], audio_feature_length, 1, 0)

        self.rgbdepth_upconvlayer1 = unet_upconv(512, 512) #1016 (audio-visual feature) = 512 (visual feature) + 504 (audio feature)
        self.rgbdepth_upconvlayer2 = unet_upconv(512, 256)
        self.rgbdepth_upconvlayer3 = unet_upconv(256, 128)
        self.rgbdepth_upconvlayer4 = unet_upconv(128, 64)
        self.rgbdepth_upconvlayer5 = unet_upconv(64, 32)
        self.rgbdepth_upconvlayer6 = unet_upconv(32, 16)
        self.rgbdepth_upconvlayer7 = unet_upconv(16, output_nc, True)

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.conv1x1(x)
        
        audio_feat = x
        
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(audio_feat)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(rgbdepth_upconv1feature)
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(rgbdepth_upconv2feature)
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(rgbdepth_upconv3feature)
        rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(rgbdepth_upconv4feature)
        rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature)
        depth_prediction = self.rgbdepth_upconvlayer7(rgbdepth_upconv6feature)
        return depth_prediction, audio_feat

class attentionNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super(attentionNet, self).__init__()
        #initialize layers
        
        self.attention_img = nn.Bilinear(512, 512, att_out_nc)
        self.attention_material = nn.Bilinear(512, 512, att_out_nc)
        self.upconvlayer1 = unet_upconv(input_nc, 512) 
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)
        self.upconvlayer4 = unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv(64, 1, True)
        
    def forward(self, rgb_feat, echo_feat, mat_feat):
        rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()
        
        attentionImg = self.attention_img(rgb_feat, echo_feat)
        attentionMat = self.attention_material(mat_feat, echo_feat)
    
        attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        # audioVisual_feature = torch.cat((rgb_feat, echo_feat), dim=1)
        
        upconv1feature = self.upconvlayer1(audioVisual_feature)
        upconv2feature = self.upconvlayer2(upconv1feature)
        upconv3feature = self.upconvlayer3(upconv2feature)
        upconv4feature = self.upconvlayer4(upconv3feature)
        attention = self.upconvlayer5(upconv4feature)
        return attention, audioVisual_feature


class PyramidattentionNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super(PyramidattentionNet, self).__init__()
        #initialize layers
        
        # self.attention_img = nn.Bilinear(512, 512, att_out_nc)
        # self.attention_material = nn.Bilinear(512, 512, att_out_nc)
        self.upconvlayer1 = unet_upconv(input_nc*2, 512) 
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)
        self.upconvlayer4 = unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv(64, 1, True)
        
    def forward(self, features):
        # rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        # echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        # mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()
        
        # attentionImg = self.attention_img(rgb_feat, echo_feat)
        # attentionMat = self.attention_material(mat_feat, echo_feat)
    
        # attentionImg = attentionImg.permute(0, 3, 1, 2).contiguous()
        # attentionMat = attentionMat.permute(0, 3, 1, 2).contiguous()
        
        # audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        
        upconv1feature = self.upconvlayer1(features)
        upconv2feature = self.upconvlayer2(upconv1feature)
        upconv3feature = self.upconvlayer3(upconv2feature)
        upconv4feature = self.upconvlayer4(upconv3feature)
        attention = self.upconvlayer5(upconv4feature)
        return attention, features


class RGBDepthNet(nn.Module):
    def __init__(self, ngf=64, input_nc=3, output_nc=1):
        super(RGBDepthNet, self).__init__()
        #initialize layers
        self.rgbdepth_convlayer1 = unet_conv(input_nc, ngf)
        self.rgbdepth_convlayer2 = unet_conv(ngf, ngf * 2)
        self.rgbdepth_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.rgbdepth_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.rgbdepth_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.rgbdepth_upconvlayer1 = unet_upconv(512, ngf * 8)
        self.rgbdepth_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.rgbdepth_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.rgbdepth_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True)
        #self.conv1x1 = create_conv(512, 8, 1, 0) #reduce dimension of extracted visual features

    def forward(self, x):
        # print("-------------------------------------------------")
        # print("IMAGE SHAPE", x.shape)
        rgbdepth_conv1feature = self.rgbdepth_convlayer1(x)
        # print("NEXT", rgbdepth_conv1feature.shape)
        rgbdepth_conv2feature = self.rgbdepth_convlayer2(rgbdepth_conv1feature)
        # print("NEXT", rgbdepth_conv2feature.shape)
        rgbdepth_conv3feature = self.rgbdepth_convlayer3(rgbdepth_conv2feature)
        # print("NEXT", rgbdepth_conv3feature.shape)
        rgbdepth_conv4feature = self.rgbdepth_convlayer4(rgbdepth_conv3feature)
        # print("NEXT", rgbdepth_conv4feature.shape)
        rgbdepth_conv5feature = self.rgbdepth_convlayer5(rgbdepth_conv4feature)
        # print("NEXT", rgbdepth_conv5feature.shape)
        
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(rgbdepth_conv5feature)
        # print("UPCONV", rgbdepth_upconv1feature.shape)
        # print("CONCATTED", torch.cat((rgbdepth_upconv1feature, rgbdepth_conv4feature), dim=1).shape)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(torch.cat((rgbdepth_upconv1feature, rgbdepth_conv4feature), dim=1))
        # print("UPCONV", rgbdepth_upconv2feature.shape)
        # print("CONCATTED", torch.cat((rgbdepth_upconv2feature, rgbdepth_conv3feature), dim=1).shape)
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(torch.cat((rgbdepth_upconv2feature, rgbdepth_conv3feature), dim=1))
        # print("UPCONV", rgbdepth_upconv3feature.shape)
        # print("CONCATTED", torch.cat((rgbdepth_upconv3feature, rgbdepth_conv2feature), dim=1).shape)
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(torch.cat((rgbdepth_upconv3feature, rgbdepth_conv2feature), dim=1))
        # print("UPCONV", rgbdepth_upconv4feature.shape)
        # print("CONCATTED", torch.cat((rgbdepth_upconv4feature, rgbdepth_conv1feature), dim=1).shape)
        depth_prediction = self.rgbdepth_upconvlayer5(torch.cat((rgbdepth_upconv4feature, rgbdepth_conv1feature), dim=1))
        # print("UPCONV", depth_prediction.shape)
        return depth_prediction, rgbdepth_conv5feature

class MaterialPropertyNet(nn.Module):
    def __init__(self, nclass, backbone):
        super(MaterialPropertyNet, self).__init__()
        
        self.pretrained = backbone
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, nclass)

    def forward(self, x):
        # pre-trained ResNet feature
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        feat = self.pretrained.layer4(x)
        x = self.pool(feat)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x, feat

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, key_in_dim, keyprojin=1066, keyprojout=1024, channelnum=1):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = channelnum , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = key_in_dim , out_channels = channelnum , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.keyproj = nn.Linear(keyprojin, keyprojout)

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, x, v):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        key_m_batchsize, key_C, key_width, key_height = v.size()
        # print("A", self.query_conv(x).shape, self.query_conv(x).view(m_batchsize,-1,width*height).shape, self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1).shape)
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        # print("B", self.key_conv(v).shape, self.key_conv(v).view(m_batchsize,-1,key_width*key_height).shape) 
        proj_key =  self.key_conv(v).view(m_batchsize,-1,key_width*key_height) # B X C x (*W*H)

        # print(proj_key.shape)

        proj_key = self.keyproj(proj_key)

        # print(proj_key.shape)
        energy =  torch.bmm(proj_query,proj_key) / math.sqrt(proj_key.shape[-1]) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        # out = self.gamma*out + x
        return out,attention
 

class Self_Attn_Downsample(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, key_in_dim, keyprojin=42*27, keyprojout=1024, channelnum=1):
        super(Self_Attn_Downsample,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = channelnum , kernel_size=2, stride=2)
        self.key_conv = nn.Conv2d(in_channels = key_in_dim , out_channels = channelnum , kernel_size=2, stride=2)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size=2, stride=2)

        self.upscale_conv = nn.ConvTranspose2d(in_channels=in_dim, out_channels=in_dim, kernel_size=2, stride=2)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.keyproj = nn.Linear(keyprojin, keyprojout)

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, x, v):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        width = width//2
        height = height//2
        key_m_batchsize, key_C, key_width, key_height = v.size()
        key_width = key_width//2
        key_height = key_height//2

        # print(self.query_conv(x).shape, self.query_conv(x).view(m_batchsize,-1,width*height).shape, self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1).shape)

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)

        # print(self.key_conv(v).shape, self.key_conv(v).view(m_batchsize,-1,key_width*key_height).shape) 

        proj_key =  self.key_conv(v).view(m_batchsize,-1,key_width*key_height) # B X C x (*W*H)

        # print(proj_key.shape)

        proj_key = self.keyproj(proj_key)

        # print(proj_key.shape)

        energy =  torch.bmm(proj_query,proj_key) / math.sqrt(proj_key.shape[-1]) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 

        # print(self.value_conv(x).shape, self.value_conv(x).view(m_batchsize,-1,width*height).shape)

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        # print(attention.shape, proj_value.shape)

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # print(out.shape)
        out = out.view(m_batchsize,C,width,height)
        # print(out.shape)

        out = self.upscale_conv(out)
        # print(out.shape)
        
        # out = self.gamma*out + x
        return out,attention


class SemanticPyramid(nn.Module):
    """
    Me and Deen's idea on self-attention at each stage of the audio and vision convolution steps
    """
    def __init__(self, audio_conv1x1_dim,
                       audio_shape,
                       audio_feature_length,
                       visual_ngf=64,
                       visual_input_nc=3,
                       output_nc=1,
                       output_semantic_nc=101,
                       vtoa=False):
        super(SemanticPyramid, self).__init__()
        # AUDIO SETUP
        self.vtoa = vtoa
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(6, 6), (4, 4), (4, 4), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(3, 3), (2, 2), (2, 2), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, 128, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        self.conv4 = create_conv(128, 256, kernel=self._cnn_layers_kernel_size[3], paddings=0, stride=self._cnn_layers_stride[3])
        self.conv5 = create_conv(256, audio_conv1x1_dim, kernel=self._cnn_layers_kernel_size[4], paddings=0, stride=self._cnn_layers_stride[4])
        # layers = [self.conv1, self.conv2, self.conv3]
        # self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(1152, audio_feature_length, 1, 0)

        # VISION SETUP

        self.rgbdepth_convlayer1 = unet_conv(visual_input_nc, visual_ngf)
        self.rgbdepth_convlayer2 = unet_conv(visual_ngf, visual_ngf * 2)
        self.rgbdepth_convlayer3 = unet_conv(visual_ngf * 2, visual_ngf * 4)
        self.rgbdepth_convlayer4 = unet_conv(visual_ngf * 4, visual_ngf * 8)
        self.rgbdepth_convlayer5 = unet_conv(visual_ngf * 8, visual_ngf * 8)
        self.rgbdepth_upconvlayer1 = unet_upconv(512, visual_ngf * 8)
        self.rgbdepth_upconvlayer2 = unet_upconv(visual_ngf * 16, visual_ngf *4)
        self.rgbdepth_upconvlayer3 = unet_upconv(visual_ngf * 8, visual_ngf * 2)
        self.rgbdepth_upconvlayer4 = unet_upconv(visual_ngf * 4, visual_ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv(visual_ngf * 2, output_nc, True)

        self.semantic_upconvlayer1 = unet_upconv(512, visual_ngf * 8)
        self.semantic_upconvlayer2 = unet_upconv(visual_ngf * 16, visual_ngf * 4)
        self.semantic_upconvlayer3 = unet_upconv(visual_ngf * 8, visual_ngf * 2)
        self.semantic_upconvlayer4 = unet_upconv(visual_ngf * 4, visual_ngf)
        self.semantic_upconvlayer5 = unet_upconv(visual_ngf * 2, output_semantic_nc, True, semanticoutput=True)


        self.attlayer1 = Self_Attn_Downsample(64, 32, keyprojin=42*27, keyprojout=1024, channelnum=4)
        self.attlayer2 = Self_Attn(128, 64, keyprojin=1066, keyprojout=1024, channelnum=16)
        self.attlayer3 = Self_Attn(256, 128, keyprojin=228, keyprojout=256, channelnum=32)
        self.attlayer4 = Self_Attn(512, 256, keyprojin=40, keyprojout=64, channelnum=64)
        # self.attlayer5 = Self_Attn(512, 64, keyprojin=18, keyprojout=16)

        if self.vtoa:
            self.vtoa_attlayer1 = Self_Attn_Downsample(32, 64, keyprojin=1024, keyprojout=1134, channelnum=4)
            self.vtoa_attlayer2 = Self_Attn(64, 128, keyprojin=1024, keyprojout=1066, channelnum=16)
            self.vtoa_attlayer3 = Self_Attn(128, 256, keyprojin=256, keyprojout=228, channelnum=32)
            self.vtoa_attlayer4 = Self_Attn(256, 512, keyprojin=64, keyprojout=40, channelnum=64)
    
    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x_audio, x_rgb):
        # BLOCK 1
        x_audio1 = self.conv1(x_audio)
        rgbdepth_conv1feature = self.rgbdepth_convlayer1(x_rgb)

        attend1, _ = self.attlayer1(rgbdepth_conv1feature, x_audio1)
        if self.vtoa:
            vtoa_attend1, _ = self.vtoa_attlayer1(x_audio1, rgbdepth_conv1feature)

        rgb_depth_conv1feature = rgbdepth_conv1feature + attend1
        if self.vtoa:
            x_audio1 = x_audio1 + vtoa_attend1

        # BLOCK 2

        x_audio2 = self.conv2(x_audio1)
        rgbdepth_conv2feature = self.rgbdepth_convlayer2(rgb_depth_conv1feature)

        attend2, _ = self.attlayer2(rgbdepth_conv2feature, x_audio2)
        if self.vtoa:
            vtoa_attend2, _ = self.vtoa_attlayer2(x_audio2, rgbdepth_conv2feature)

        rgb_depth_conv2feature = rgbdepth_conv2feature + attend2
        if self.vtoa:
            x_audio2 = x_audio2 + vtoa_attend2

        # BLOCK 3

        x_audio3 = self.conv3(x_audio2)
        rgbdepth_conv3feature = self.rgbdepth_convlayer3(rgb_depth_conv2feature)

        attend3, _ = self.attlayer3(rgbdepth_conv3feature, x_audio3)
        if self.vtoa:
            vtoa_attend3, _ = self.vtoa_attlayer3(x_audio3, rgbdepth_conv3feature)

        rgb_depth_conv3feature = rgbdepth_conv3feature + attend3
        if self.vtoa:
            x_audio3 = x_audio3 + vtoa_attend3

        # BLOCK 4

        x_audio4 = self.conv4(x_audio3)
        rgbdepth_conv4feature = self.rgbdepth_convlayer4(rgb_depth_conv3feature)

        attend4, _ = self.attlayer4(rgbdepth_conv4feature, x_audio4)
        if self.vtoa:
            vtoa_attend4, _ = self.vtoa_attlayer4(x_audio4, rgbdepth_conv4feature)

        rgb_depth_conv4feature = rgbdepth_conv4feature + attend4
        if self.vtoa:
            x_audio4 = x_audio4 + vtoa_attend4

        # BLOCK 5

        x_audio5 = self.conv5(x_audio4)
        rgb_depth_conv5feature = self.rgbdepth_convlayer5(rgb_depth_conv4feature)

        # attend5, _ = self.attlayer5(rgb_depth_conv5feature, x_audio5)
        # rgb_depth_conv5feature = rgb_depth_conv5feature + attend5

        audio_feat_flat = x_audio5.view(x_audio5.shape[0], -1, 1, 1)

        audio_feat_flat = self.conv1x1(audio_feat_flat)
        audio_feat_repeated = audio_feat_flat.repeat(1, 1, rgb_depth_conv5feature.shape[-2], rgb_depth_conv5feature.shape[-1])

        audioVisual_feature = rgb_depth_conv5feature + audio_feat_repeated

        # UPCONV BLOCK

        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(audioVisual_feature)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(torch.cat((rgbdepth_upconv1feature, rgbdepth_conv4feature), dim=1))
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(torch.cat((rgbdepth_upconv2feature, rgbdepth_conv3feature), dim=1))
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(torch.cat((rgbdepth_upconv3feature, rgbdepth_conv2feature), dim=1))
        depth_prediction = self.rgbdepth_upconvlayer5(torch.cat((rgbdepth_upconv4feature, rgbdepth_conv1feature), dim=1))

        # SEMANTIC UPCONV BLOCK

        # print("LETS GOOOOOO")
        # print("features", audioVisual_feature.shape)
        sem1feature = self.semantic_upconvlayer1(audioVisual_feature)
        # print(sem1feature.shape)
        sem2feature = self.semantic_upconvlayer2(torch.cat((sem1feature, rgbdepth_conv4feature), dim=1))
        # print(sem2feature.shape)
        sem3feature = self.semantic_upconvlayer3(torch.cat((sem2feature, rgbdepth_conv3feature), dim=1))
        # print(sem3feature.shape)
        sem4feature = self.semantic_upconvlayer4(torch.cat((sem3feature, rgbdepth_conv2feature), dim=1))
        # print(sem4feature.shape)
        sem5feature = self.semantic_upconvlayer5(torch.cat((sem4feature, rgbdepth_conv1feature), dim=1))
        # print(sem5feature.shape)

        return depth_prediction, sem5feature


class SimpleAudioMultiviewFeatDepthNet(nn.Module):
    ## strucure adapted from VisualEchoes [ECCV 2020]
    r"""A Simple 3-Conv CNN followed by a fully connected layer
    """
    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, output_nc=1):
        super(SimpleAudioMultiviewFeatDepthNet, self).__init__()
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, conv1x1_dim, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        layers = [self.conv1, self.conv2, self.conv3]
        self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(conv1x1_dim * cnn_dims[0] * cnn_dims[1], audio_feature_length, 1, 0)

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x):
        # print("audio shape before feature", x.shape)
        x = self.feature_extraction(x)
        # print("audio shape after feature", x.shape)
        # x = x.view(x.shape[0], -1, 1, 1)
        # print("audio shape after view", x.shape)
        # x = self.conv1x1(x)
        # print("audio shape 1x1", x.shape)
        
        audio_feat = x
        return  audio_feat


class SimpleAudioMultiviewDepthNet(nn.Module):
    ## strucure adapted from VisualEchoes [ECCV 2020]
    r"""A Simple 3-Conv CNN followed by a fully connected layer
    """
    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, output_nc=1):
        super(SimpleAudioMultiviewDepthNet, self).__init__()
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        # self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        # self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        # self.conv3 = create_conv(64, conv1x1_dim, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        # layers = [self.conv1, self.conv2, self.conv3]
        # self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(conv1x1_dim * cnn_dims[0] * cnn_dims[1], audio_feature_length, 1, 0)

        self.rgbdepth_upconvlayer1 = unet_upconv(512, 512) #1016 (audio-visual feature) = 512 (visual feature) + 504 (audio feature)
        self.rgbdepth_upconvlayer2 = unet_upconv(512, 256)
        self.rgbdepth_upconvlayer3 = unet_upconv(256, 128)
        self.rgbdepth_upconvlayer4 = unet_upconv(128, 64)
        self.rgbdepth_upconvlayer5 = unet_upconv(64, 32)
        self.rgbdepth_upconvlayer6 = unet_upconv(32, 16)
        self.rgbdepth_upconvlayer7 = unet_upconv(16, output_nc, True)

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x):
        
        
        audio_feat = x
        
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(audio_feat)
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(rgbdepth_upconv1feature)
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(rgbdepth_upconv2feature)
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(rgbdepth_upconv3feature)
        rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(rgbdepth_upconv4feature)
        rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature)
        depth_prediction = self.rgbdepth_upconvlayer7(rgbdepth_upconv6feature)
        return depth_prediction