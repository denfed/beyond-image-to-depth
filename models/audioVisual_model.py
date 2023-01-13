import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from torch.autograd import Variable
import torch.nn as nn

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_rgbdepth, self.net_audio, self.net_attention, self.net_material = nets
        

    def forward(self, input, volatile=False):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']

        audio_depth, audio_feat = self.net_audio(audio_input)
        # print(audio_depth.shape, audio_feat.shape)
        img_depth, img_feat = self.net_rgbdepth(rgb_input)
        material_class, material_feat = self.net_material(rgb_input)
        audio_feat = audio_feat.repeat(1, 1, img_feat.shape[-2], img_feat.shape[-1]) #tile audio feature
        alpha, _ = self.net_attention(img_feat, audio_feat, material_feat)
        depth_prediction = ((alpha*audio_depth)+((1-alpha)*img_depth)) 

        
        output =  {'img_depth': img_depth * self.opt.max_depth,
                    'audio_depth': audio_depth * self.opt.max_depth,
                    'depth_predicted': depth_prediction * self.opt.max_depth, 
                    'attention': alpha,
                    'img': rgb_input,
                    'audio': audio_input,
                    'depth_gt': depth_gt}
        return output


class AudioOnlyModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioOnlyModel, self).__init__()
        self.opt = opt

        self.net_audio = nets

    def forward(self, input, volatile=False):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']

        # Add the model stuff here
        depth_prediction, _ = self.net_audio(audio_input)

        output = {'depth_predicted': depth_prediction * self.opt.max_depth,
                  'img': rgb_input,
                  'audio': audio_input,
                  'depth_gt': depth_gt}
        
        return output


class AudioVisualPyramidAttentionModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualPyramidAttentionModel'

    def __init__(self, net, opt):
        super(AudioVisualPyramidAttentionModel, self).__init__()
        self.opt = opt

        self.net_pyramid = net

    def forward(self, input, volatile=False):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']

        # Add the model stuff here
        depth_prediction = self.net_pyramid(audio_input, rgb_input)

        output = {'depth_predicted': depth_prediction * self.opt.max_depth,
                  'img': rgb_input,
                  'audio': audio_input,
                  'depth_gt': depth_gt}
        
        return output


class AudioVisualPyramidAttentionAudioDepthModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualPyramidAttentionModel'

    def __init__(self, net, opt):
        super(AudioVisualPyramidAttentionAudioDepthModel, self).__init__()
        self.opt = opt

        self.net_pyramid, self.net_attention = net

    def forward(self, input, volatile=False):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']

        # Add the model stuff here
        vision_depth_prediction, audio_depth_prediction, features = self.net_pyramid(audio_input, rgb_input)
        alpha, _ = self.net_attention(features)
        depth_prediction = ((alpha*audio_depth_prediction)+((1-alpha)*vision_depth_prediction)) 

        output = {'depth_predicted': depth_prediction * self.opt.max_depth,
                  'audio_depth_predicted': audio_depth_prediction * self.opt.max_depth,
                  'vision_depth_predicted': vision_depth_prediction * self.opt.max_depth,
                  'attention': alpha,
                  'img': rgb_input,
                  'audio': audio_input,
                  'depth_gt': depth_gt}
        
        return output

class SemanticPyramidModel(torch.nn.Module):
    def name(self):
        return 'SemanticPyramidModel'

    def __init__(self, net, opt):
        super(SemanticPyramidModel, self).__init__()
        self.opt = opt
        self.net_semantic = net

    def forward(self, input):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']
        semantic_gt = input['semantic']

        depth_prediction, semantic_prediction = self.net_semantic(audio_input, rgb_input)

        output = {'depth_predicted': depth_prediction * self.opt.max_depth,
                  'semantic_predicted': semantic_prediction,
                  'img': rgb_input,
                  'audio': audio_input,
                  'depth_gt': depth_gt,
                  'semantic_gt': semantic_gt}
        
        return output


class AudioVisualMultiviewModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualMultiviewModel'

    def create_conv(self, input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
        model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
        if(batch_norm):
            model.append(nn.BatchNorm2d(output_channels))
        if(Relu):
            model.append(nn.ReLU())
        return nn.Sequential(*model)

    def __init__(self, nets, opt):
        super(AudioVisualMultiviewModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_rgbdepth, self.net_audio_feat, self.net_audio_depth, self.net_attention, self.net_material = nets

        self.multiview_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.multiview_projection = nn.Linear(512+4, 128)

        self.featurewise=1

        if self.featurewise:
            self.attention_softmax = nn.Softmax(dim=1)
        else:
            self.attention_softmax = nn.Softmax(dim=2)

        self.conv1x1 = self.create_conv(60928, 512, 1, 0)
        
    def forward(self, input, volatile=False):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']
        rgb_multiview = input['multiview']
        orientation = input['orientation']
        metadata = input['or_metadata']
        # print("ORIENTATION", orientation, orientation.shape)
        # print("RGB MULTIVIEW", rgb_multiview.shape)

        b, mv, c, h, w = rgb_multiview.size()

        # print("VIEWED SAHPE", rgb_multiview.view(b*mv, c, h, w).shape)

        _, multiview_feat = self.net_rgbdepth(rgb_multiview.view(b*mv, c, h, w))
        # print("MULTIVIEW SHAPE", multiview_feat.shape)

        b_ex, c_f, h_f, w_f = multiview_feat.size()

        multiview_feat = multiview_feat.view(b, mv, c_f, h_f, w_f)
        # print("MULTIVIEW SHAPE after view", multiview_feat.shape)
        multiview_feat = self.multiview_avgpool(multiview_feat).squeeze(-1).squeeze(-1)
        # print("multivew features", multiview_feat.shape)
        # print(multiview_feat)

        # print(metadata.shape)

        multiview_feat = torch.cat((metadata, multiview_feat), dim=2)
        # print(orientation)
        # print(multiview_feat)
        # print(multiview_feat[0], orientation[0])
        # print(multiview_feat[1], orientation[1])

        # print("MULTIVIEW SHAPE after global average pooling", multiview_feat.shape)

        multiview_feat = self.multiview_projection(multiview_feat).unsqueeze(-1).unsqueeze(-1)
        multiview_feat = nn.functional.normalize(multiview_feat, dim=2)

        # print("MULTIVIEW SHAPE after projection", multiview_feat.shape)

        audio_feat = self.net_audio_feat(audio_input)
        audio_feat_normalized = nn.functional.normalize(audio_feat, dim=1)
# 
        # print("SEPARATED AUDIO FEAT", audio_feat.shape)

        attention = torch.einsum('nchw,nmcqa->nmhw', [audio_feat_normalized, multiview_feat])
        # print("ATTENTION", attention.shape)

        attention = attention.view(b, mv, -1)
        # print("ATTENTION VIEW", attention.shape)
        
        if self.featurewise:
            attention = self.attention_softmax(attention)
            # print(attention)

            # att_at_orientation = attention[:, orientation, :]
            att_at_orientation = attention[torch.arange(attention.size(0)), orientation]*4
            # print(att_at_orientation, att_at_orientation.shape)


            # print("multiply these", att_at_orientation.shape, audio_feat.shape)
            # audio_depth, audio_feat = self.net_audio(audio_input)
            audio_feat = audio_feat.view(b, 128, -1)
            # print("multiply these", att_at_orientation.unsqueeze(1).shape, audio_feat.shape)
            attended_audio_feat = att_at_orientation.unsqueeze(1)*audio_feat
        else: 
            # Softmax across all features for each view, i.e. 4 softmax operations
            attention = self.attention_softmax(attention)
            
            att_at_view = attention[torch.arange(attention.size(0)), orientation]
            # print("attention at view:", att_at_view, att_at_view.shape)

            mask = torch.ones(b, 4).cuda()
            # print(mask)
            mask.scatter_(1, orientation.unsqueeze(1), 0.)
            # print(mask, mask.shape)

            att_masked_otherviews = attention * mask.unsqueeze(-1)

            att_masked_otherviews = (att_masked_otherviews*-1)/3

            # print(att_masked_otherviews, att_masked_otherviews.shape)

            att_masked_otherviews = torch.sum(att_masked_otherviews, dim=1)

            # print(att_masked_otherviews, att_masked_otherviews.shape, "summed together")

            att_at_view = att_at_view - att_masked_otherviews

            audio_feat = audio_feat.view(b, 128, -1)

            attended_audio_feat = att_at_view.unsqueeze(1)*audio_feat

            # att_at_otherviews = torch.cat((attention[torch.arange(attention.size(0)), attention[:orientation], x[orientation+1:]]))
            # print("attention at 3 other views:", att_at_otherviews, att_at_otherviews.shape)
            #row_exclude = 2
            #x = torch.cat((x[:row_exclude],x[row_exclude+1:]))

        attended_audio_feat = attended_audio_feat.view(attended_audio_feat.shape[0], -1, 1, 1)
        # print("EXTRA audio shape after view", attended_audio_feat.shape)
        attended_audio_feat = self.conv1x1(attended_audio_feat)
        # print("EXTRA audio shape 1x1", attended_audio_feat.shape)

        audio_feat = attended_audio_feat


        audio_depth = self.net_audio_depth(audio_feat)

        # print(audio_depth.shape, audio_feat.shape)
        img_depth, img_feat = self.net_rgbdepth(rgb_input)
        material_class, material_feat = self.net_material(rgb_input)
        audio_feat = audio_feat.repeat(1, 1, img_feat.shape[-2], img_feat.shape[-1]) #tile audio feature
        alpha, _ = self.net_attention(img_feat, audio_feat, material_feat)
        depth_prediction = ((alpha*audio_depth)+((1-alpha)*img_depth)) 
        # depth_prediction = audio_depth

        
        output =  {'img_depth': img_depth * self.opt.max_depth,
                    'audio_depth': audio_depth * self.opt.max_depth,
                    'depth_predicted': depth_prediction * self.opt.max_depth, 
                    'attention': alpha,
                    'img': rgb_input,
                    'audio': audio_input,
                    'depth_gt': depth_gt}

        return output