#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 06:00:09 2019

@author: aneesh
"""

import functools
import math
import torch.nn.functional as F
from networks.small_lsk import LSKsmallBlock
from functools import partial
from networks.VRAttention import VRAttention
from networks.SRM import SRMLayer
import torch
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
import sys
import os
import torch
import torch.nn as nn

from timm.models.layers import DropPath
import math
from functools import partial
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

from mmcv.cnn import build_norm_layer
import numpy as np
from networks.CLMGNet import EPSABlock

nonlinearity = partial(F.relu, inplace=True)


class SE_Module(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.Sigmoid()
        )
        self.activation = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class ASPPConv(nn.Sequential):
    """
        ASPP卷积模块的定义
        """

    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """
        ASPP的pooling层
        """

    def __init__(self, in_channels, out_channels):  # [in_channel=out_channel=256]
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # [256*1*1]
            # 自适应平均池化层，只需要给定输出的特征图的尺寸(括号内数字)就好了
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        # x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
        ASPP空洞卷积块
        """

    def __init__(self, in_channels_1, in_channels_2, out_channels):  # atrous_rates=(6, 12, 18)
        super(ASPP, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_1, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),  # [256*64*64]
            nn.ReLU())  # 1x1卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels_2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),  # [256*64*64]
            nn.ReLU())  # 1x1卷积
        # rate1, rate2, rate3 = tuple(atrous_rates)
        # self.conv2 = (ASPPConv(in_channels, out_channels, rate1))   # 3*3卷积( padding=6, dilation=6 )
        # self.conv3 = (ASPPConv(in_channels, out_channels, rate2))   # 3*3 卷积( padding=12, dilation=12 )
        # self.conv4 = (ASPPConv(in_channels, out_channels, rate3))   # 3*3 卷积( padding=18, dilation=18 )  [256*64*64]
        # self.conv5 = (ASPPPooling(in_channels, out_channels))       # 全局平均池化操作，输出尺寸为（1,1） [256*1*1]

        # GFF部分

        self.conva = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
            ,
            nn.BatchNorm2d(1)
        )
        self.convb = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
            nn.BatchNorm2d(1)
        )
        self.project = nn.Sequential(  # 特征融合,此时输入通道是原始输入通道的5倍。输出的结果又回到原始的通道数。
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),  # [1280*64*64]
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x1, x2):
        x1 = self.conv1(x1)  # torch.Size([2, 64, 64, 64])
        x = self.conv2(x2)
        # print('x2',x.shape)#torch.Size([2, 64, 32, 32])
        x2 = F.interpolate(x, size=(2 * x.shape[2], 2 * x.shape[3]), mode='bilinear',
                           align_corners=False)
        # print('x2x2',x2.shape)#torch.Size([2, 64, 64, 64])
        g1 = self.conva(x1)  # torch.Size([2, 1, 64, 64])
        # print('g1',g1.shape)
        g2 = self.convb(x2)  # torch.Size([2, 1, 64, 64])
        # print('g2',g2.shape)
        # g3 = self.convc(c)#(b, 1 ,h, w)
        # g4 = self.convd(d)#(b, 1 ,h, w)
        # g5 = self.conve(e)#(b, 1 ,h, w)
        # print('e',e.shape)
        # print('g5',g5.shape)

        a_gff = (1 + g1) * x1 + (1 - g1) * (g2 * x2)  # a_gff torch.Size([1, outchannels, 128, 128])
        # a_gff = self.convaa(a_gff) + a_gff

        b_gff = (1 + g2) * x2 + (1 - g2) * (g1 * x1)
        # b_gff = self.convbb(b_gff) + b_gff

        # c_gff = (1 + g3) * c + (1 - g3) * (g2 * b + g4 * d + g5 * e + g1 * a)
        # c_gff = self.convcc(c_gff) + c_gff

        # d_gff = (1 + g4) * d + (1 - g4) * (g2 * b + g3 * c + g5 * e + g1 * a)
        # d_gff = self.convdd(d_gff) + d_gff

        # e_gff = (1 + g5) * e + (1 - g5) * (g2 * b + g3 * c + g4 * d + g1 * a)
        # e_gff = self.convee(e_gff) + e_gff

        gff_outs = torch.cat([a_gff, b_gff], dim=1)  # torch.Size([1, outchannels*5, 128, 128])

        output = self.project(gff_outs)
        return output


class ChannelAttention(nn.Module):
    def __init__(self, Channel_nums):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max  # torch.Size([2, 3, 1, 1])
        # print('F_add',F_add.shape)
        F_add_ = F_add.squeeze(-1).permute(0, 2, 1)  # torch.Size([2, 1, 3])
        # print('self.k',(self.k))
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1)
        out = self.sigmoid(F_add_)
        return out


class SpatialAttention_SSGD(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_SSGD, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out1 = self.conv1(out)
        # out1 = self.bn1(out1)
        out2 = self.relu1(out1)
        out = self.sigmoid(out2)

        y = x * out.view(out.size(0), 1, out.size(-2), out.size(-1))
        # y = y + residual
        return y


class SpatialAttention(nn.Module):
    def __init__(self, Channel_num):
        super(SpatialAttention, self).__init__()
        self.channel = Channel_num
        self.Lambda = 0.6  # separation rate
        self.C_im = self.get_important_channelNum(Channel_num)  # 0.6
        self.C_subim = Channel_num - self.C_im  # 0.4
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.SpatialAttention_SSGD = SpatialAttention_SSGD()

    def get_important_channelNum(self, C):  # even|t|最近偶数
        t = self.Lambda * C
        floor = math.floor(t)
        C_im = floor + floor % 2
        return C_im  # 返回值一定是偶数

    def get_im_subim_channels(self, C_im, M):
        _, topk = torch.topk(M, dim=1, k=C_im)  # 降序得到前C_im（总数的0.6）个索引
        important_channels = torch.zeros_like(M)  # M形状为(b,c,1,1))
        # subimportant_channels = torch.ones_like(M)

        important_channels = important_channels.scatter(1, topk, 1)  # 前C_im大的是1，剩余的是0
        # subimportant_channels = subimportant_channels.scatter(1, topk, 0) #前C_im大的是0，剩余的是1 及就是：不重要的部分是1
        return important_channels

    def get_features(self, im_channels, channel_refined_feature):
        import_features = im_channels * channel_refined_feature
        return import_features

    def forward(self, x, M):  # (b,c,h,w)和(b,c,1,1)
        important_channels = self.get_im_subim_channels(self.C_im, M)  # (0.6*c ,(b,c,1,1))
        important_features = self.get_features(important_channels, x)  # 和mask相乘得到F1‘和 F2’

        SpatialAttention_SSGD_Output = self.SpatialAttention_SSGD(important_features)

        # im_AvgPool = torch.mean(SpatialAttention_SSGD_Output, dim=1, keepdim=True) * (self.channel / self.C_im)
        # im_MaxPool, _ = torch.max(SpatialAttention_SSGD_Output, dim=1, keepdim=True)
        #
        # im_x = torch.cat([im_AvgPool, im_MaxPool], dim=1)
        # A_S1 = self.norm_active(self.conv(im_x))
        # F1 = important_features * A_S1
        # F1 = F1 + important_features

        return SpatialAttention_SSGD_Output


class ResBlock_HAM(nn.Module):
    def __init__(self, Channel_nums):
        super(ResBlock_HAM, self).__init__()
        self.channel = Channel_nums
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        # self.ChannelAttention = ChannelAttention(self.channel)
        self.SpatialAttention = SpatialAttention(self.channel)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        # residual = x_in
        # channel_attention_map = self.AdaptiveAvgPool2d(x_in) #(b,c,1,1)
        channel_attention_map = self.AdaptiveAvgPool2d(x_in)  # (b,c,1,1)
        # channel_refined_feature = channel_attention_map * x_in #(b,c,w,h)
        final_refined_feature = self.SpatialAttention(x_in, channel_attention_map)
        out = self.relu(final_refined_feature)
        return out


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


def SplitChannels(channels, kernel_num):
    split_channels = [channels // kernel_num for _ in range(kernel_num)]
    # 将channels//kernel_num这个结果循环kernel_num次 放入一个数组中
    # for _ in range(n)中的'_' 是占位符， 表示不在意变量的值，只是用于循环遍历n次；
    # print(split_channels)
    split_channels[0] += channels - sum(split_channels)  #
    # print(split_channels)
    return split_channels


# SplitChannels(8,4)
class ASSP_res2net_dilation(nn.Module):
    def __init__(self, channels, kernel_num=4):
        # spectral 光谱带数量
        # 第一次调用 MFC(8, 24, (5, 3, 3), 4)
        super(ASSP_res2net_dilation, self).__init__()
        # self.rate = 1.0
        # self.before_ASSP = Conv1x1BNReLU(channels, int(channels * (self.rate)))
        # self.before_ASSP = DepthWiseConv(channels, int(channels * (self.rate)))
        self.channels = channels

        self.kernel_num = kernel_num
        # self.k1 = kernel_size[0]
        # self.k2 = kernel_size[1]
        # self.k3 = kernel_size[2]

        self.sp = SplitChannels(self.channels, self.kernel_num)  # 做分割 :[2, 2, 2, 2][5, 5, 5, 5] [14, 12, 12, 12]
        dilations = [1, 2, 3, 4]
        self.conv1_1 = assp_branch(self.sp[0], self.sp[0], 3, dilation=dilations[0])
        self.conv1_2 = assp_branch(self.sp[1], self.sp[1], 3, dilation=dilations[1])
        self.conv1_3 = assp_branch(self.sp[2], self.sp[2], 3, dilation=dilations[2])
        # self.conv1_4 = assp_branch(self.sp[3], self.sp[3], 3, dilation=dilations[3])
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv2d(in_channels, in_channels//4, 1, bias=False),
            nn.BatchNorm2d(self.sp[3]),
            nn.ReLU(inplace=True))

        # self.after_ASSP = Conv1x1BNReLU(int(channels * (self.rate)),channels)
        # self.after_ASSP = DepthWiseConv(int(channels * (self.rate)),channels)

    def forward(self, x):
        # if self.rate != 1.0:
        #     x = self.before_ASSP(x)

        x_split = torch.split(x, self.sp, dim=1)

        x_1_1 = self.conv1_1(x_split[0])
        x_1_2 = self.conv1_2(x_split[1] + x_1_1)
        x_1_3 = self.conv1_3(x_split[2] + x_1_2)
        # x_1_4 = self.conv1_4(x_split[3])
        x_1_4 = F.interpolate(self.avg_pool(x_split[3] + x_1_3), size=(x.size(2), x.size(3)), mode='bilinear',
                              align_corners=True)
        x = torch.cat([x_1_1, x_1_2, x_1_3, x_1_4], dim=1)
        # if self.rate != 1.0:
        #     x = self.after_ASSP(x)
        return x


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)


use_sync_bn = False


def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True


def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            if self.kernel_size == 19:
                self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=in_channels, kernel_size=7,
                                          stride=1, padding=9, dilation=3, groups=in_channels)
            else:
                self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                          stride=stride, padding=small_kernel // 2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)  # 这部分应该只是搭建一下训练时的模型
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            if self.kernel_size == 19:
                eq_k += nn.functional.pad(small_k, [(7 - self.small_kernel) // 2] * 4)

            else:
                eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                      out_channels=self.lkb_origin.conv.out_channels,
                                      kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                      padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                      groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


# class DWConv(nn.Module):
#     def __init__(self, dim):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=1)
#         self.weight = self.dwconv.weight
#         self.bias = self.dwconv.bias
#         self.dim = 1
#
#     def forward(self, x):
#         x = self.dwconv(x)
#         return x


def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


class GRNwithNCWH(nn.Module):  # 全局响应归一化
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))  # 在之前已经把通道放到了最后一个维度，用于缩放归一化后的特征图
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))  # 用于平移归一化后的特征图

    def forward(self, x):
        # p=2表示是L2范数
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)     # 输入特征图 x 在高度（H）和宽度（W）维度上计算L2范数，每个值的平方和开根号
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)      # 对L2范数进行归一化，非标准的一种归一化
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta      # (self.gamma * Nx + 1) * x 对输入 x 进行缩放操作
        else:
            return (self.gamma * Nx + 1) * x


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class PMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        # self.fc1 = nn.Conv2d(in_features, hidden_features, 1, groups=hidden_features)       # bias=True
        ####################################
        self.mlp_1x1_kxk = nn.Sequential()
        self.mlp_1x1_kxk.add_module('conv1',
                                    nn.Conv2d(in_channels=in_features, out_channels=hidden_features,
                                              kernel_size=1, stride=1, padding=0, groups=hidden_features, bias=False))
        self.mlp_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=1, num_features=hidden_features,
                                                         affine=True))
        self.mlp_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features,
                                                       kernel_size=3, stride=1, padding=0, groups=hidden_features,
                                                       bias=False))
        self.mlp_1x1_kxk.add_module('bn2', nn.BatchNorm2d(hidden_features))
        ####################################
        # groups=hidden_features
        # self.conv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=True)
        self.act = act_layer()
        # self.act = nn.Sequential(nn.GELU(), GRNwithNCWH(hidden_features, use_bias=True))
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, groups=hidden_features)      # bias=True
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.BatchNorm2d(out_features)
        layer_scale_init_value = 1e-2
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(out_features), requires_grad=True)
        self.reparam = False

    def forward(self, x):
        if self.reparam:
            # x = self.fc1(x)
            x = self.mlp_1x1_kxk(x)
            x = self.act(x)
            x = self.drop(x)
            # x = self.drop(x)
            x = self.fc2(x)
            # print('mlp has reparamed')

        else:
            # x = self.fc1(x)
            # x = self.conv(x)   # 这里必须是CONVgroup为1或者fc1能变为深度卷积，不然无法重参数化
            x = self.mlp_1x1_kxk(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = self.norm2(x)
            x = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x

        return x

    def merge_kernel(self):
        same_scale = self.layer_scale_2.data
        new_fc1 = nn.Conv2d(self.fc2.in_channels, self.fc2.out_channels, 1, groups=self.fc2.groups)
        new_fc1.weight.data = self.fc2.weight
        new_fc1.bias.data = self.fc2.bias
        # new_fc1.weight.data = self.mlp.fc2.weight * same_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # new_fc1.bias.data = self.mlp.fc2.bias * same_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        k, b = fuse_bn(new_fc1, self.norm2)
        k = k * same_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b = b * same_scale
        new_fc2 = nn.Conv2d(self.fc2.in_channels, self.fc2.out_channels, 1, groups=self.fc2.groups)
        new_fc2.weight.data = k
        new_fc2.bias.data = b
        self.fc2 = new_fc2
        k_1x1_kxk_first = self.mlp_1x1_kxk.conv1
        k_1x1_kxk_first, b_1x1_kxk_first = fuse_bn(k_1x1_kxk_first, self.mlp_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = fuse_bn(self.mlp_1x1_kxk.conv2, self.mlp_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.hidden_features)
        new_conv = nn.Conv2d(self.in_features, self.hidden_features, kernel_size=3, padding=1, groups=self.hidden_features, bias=True)
        new_conv.weight.data = k_1x1_kxk_merged
        new_conv.bias.data = b_1x1_kxk_merged
        self.mlp_1x1_kxk = new_conv
        self.reparam = True


# 此为图中的FFN，不改变通道数和图像大小
class SMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)       # bias=True
        # groups=hidden_features
        self.conv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=True)
        self.act = act_layer()
        # self.act = nn.Sequential(nn.GELU(), GRNwithNCWH(hidden_features, use_bias=True))
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)      # bias=True
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.BatchNorm2d(out_features)
        layer_scale_init_value = 1e-2
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(out_features), requires_grad=True)
        self.reparam = False

    def forward(self, x):
        if self.reparam:
            x = self.fc1(x)
            x = self.conv(x)
            x = self.act(x)
            x = self.drop(x)
            # x = self.drop(x)
            x = self.fc2(x)
            # print('mlp has reparamed')

        else:
            x = self.fc1(x)
            x = self.conv(x)   # 这里必须是CONVgroup为1或者fc1能变为深度卷积，不然无法重参数化
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = self.norm2(x)
            x = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x

        return x

    def merge_kernel(self):
        same_scale = self.layer_scale_2.data
        new_fc1 = nn.Conv2d(self.fc2.in_channels, self.fc2.out_channels, 1)
        new_fc1.weight.data = self.fc2.weight
        new_fc1.bias.data = self.fc2.bias
        # new_fc1.weight.data = self.mlp.fc2.weight * same_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # new_fc1.bias.data = self.mlp.fc2.bias * same_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        k, b = fuse_bn(new_fc1, self.norm2)
        k = k * same_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b = b * same_scale
        new_fc2 = nn.Conv2d(self.fc2.in_channels, self.fc2.out_channels, 1)
        new_fc2.weight.data = k
        new_fc2.bias.data = b
        self.fc2 = new_fc2
        self.reparam = True

    # def merge_kernel(self):
        # k, b = transIII_1x1_kxk(self.fc1.weight, self.fc1.bias, self.conv.weight, self.conv.bias, groups=self.conv.groups)
        # dwconv_weight_data = F.conv2d(self.dwconv.weight, self.fc1.weight.permute(1, 0, 2, 3))
        # b_hat = (self.dwconv.weight * self.dwconv.bias.reshape(1, -1, 1, 1)).sum((1, 2, 3))
        # new_conv = nn.Conv2d(self.conv.in_channels, self.conv.in_channels, 3, 1, 1, bias=True, groups=self.conv.groups)
        # new_conv.weight.data = k
        # new_conv.bias.data = b
        # self.conv = new_conv


def channel_softpooling(input_tensor, epsilon=1e-10):
    input_tensor = torch.clamp(input_tensor, min=epsilon)

    log_values = torch.log(input_tensor)
    total_log_sum = torch.sum(log_values, dim=1, keepdim=True)
    # 避免 total_log_sum 为零的情况
    total_log_sum = torch.clamp(total_log_sum, min=epsilon)

    weights = log_values / total_log_sum
    weighted_sum = torch.sum(input_tensor * weights, dim=1, keepdim=True)

    return weighted_sum


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)   # view方法中的 -1 表示自动推断该维度的大小


class ALKC(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.norm1 = LayerNorm(d_model, eps=1e-6, data_format='channels_first')
        self.ASAttention = ASA(d_model)
        # self.dwsmallkernel = get_dwconv(d_model, 19, False, 9, 1)
        # self.norm2 = nn.BatchNorm2d(d_model)
        # self.dwsmallkernel = DilatedReparamBlock(d_model, 19, False)
        # self.spatial_gating_unit1 = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.norm1(x)
        # x = self.dwsmallkernel(x)
        # x = self.norm2(x)
        x = self.ASAttention(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


def get_bn_dilated(dim, use_sync_bn=False):
    if use_sync_bn:      # 在使用多 GPU 或分布式训练时。它能够更有效地处理分布式环境中的批标准化操作
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def get_conv2d_dilated(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               # 如果kernel_size 是一个元组，比如 (3, 5)，则表示卷积核的大小为 (3, 5),
               # 如果padding是一个元组，则第一个元素表示垂直方向（上下）的填充数量，第二个元素表示水平方向（左右）的填充数量。
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)      # 转换成元组，迭代两次（数字不是可迭代对象）
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)   # 是元组就不用迭代两次
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def fuse_bn_nondilated(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std


# convert_dilated_to_nondilated(dilated_kernel, dilated_r)
# 注意卷积核的shape！！！！
#第一个维度表示输出通道数 out_channels，接下来的两个维度表示输入通道数 in_channels 和卷积核的高度和宽度 kernel_size。（最后一个维度是卷积核的深度，即指卷积核是二维的）
# 输入通道（Input Channels）：
# 输入通道数是指卷积核与输入特征图匹配的通道数。例如，如果输入是一张RGB图像，那么输入通道数就是3，因为RGB图像有3个颜色通道（红色、绿色、蓝色）。
# 输出通道（Output Channels）：
# 输出通道数是指卷积操作后生成的特征图的通道数。这个数值由卷积核的数量决定。每个卷积核会生成一个单独的输出特征图（或通道）。
def convert_dilated_to_nondilated(kernel, dilate_rate):               # 需要
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    # kernel.size(1) == 1表示输入卷积核是一个深度可分离卷积
    if kernel.size(1) == 1:                                  # kernel.size(1) == 1 表示输入卷积核是一个深度可分离卷积（Depthwise Convolution）核，因为在深度可分离卷积中，输入通道数 in_channels 通常为 1
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):       # 遍历这个核的各个通道
            # 转置卷积，相当于把空洞点置为0
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)  # 转置卷积的意思就是现在把核作为identity_kernel然后用identity_kernel来卷kernel[:,i:i+1,:,:]，步长为空洞率，所以正好跳过了空洞点，并且保留了非空洞点，相当于把空洞点置为0
            slices.append(dilated)
        return torch.cat(slices, dim=1)


# 需要
def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):   # dilated_r是空洞率，large_kernel是最大的核和bn层融合后的等效核，dilated_kernel小核的等效核
    large_k = large_kernel.size(2)   # 高度
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1       # 这就跟自己计算感受野的公式一样
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


# 需要
class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):            # 默认deploy=False
        super().__init__()
        self.lk_origin = get_conv2d_dilated(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)                              #  attempt_use_lk_impl=True是用来加速的
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 29:
            self.kernel_sizes = [5, 9, 19, 21, 25]
            self.dilates = [1, 2, 1, 1, 1]
        elif kernel_size == 19:
            self.kernel_sizes = [5, 9]
            self.dilates = [1, 2]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [3, 5]
            self.dilates = [1, 1]
            # self.kernel_sizes = [3, 5]
            # self.dilates = [1, 1]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:         # 不考虑有BN的时候
            self.origin_bn = get_bn_dilated(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                # 在当前对象中设置一个名为 'dil_conv_k{}_{}'.format(k, r) 的属性，属性的值是后面给出的 nn.Conv2d 对象
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))  # // 2是取整操作，padding这么取是为了让输入输出大小相同
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn_dilated(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode  # 融合之后就直接走这条路了
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))  # lk_origin是图中原k*k大小的核
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):            # merge的开关，在test中使用
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn_nondilated(self.lk_origin, self.origin_bn)   # 这是融合最大的核和bn层
            for k, r in zip(self.kernel_sizes, self.dilates):          # 这是通过循环相加搞定融合的参数
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))      # 每一个小核
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))          # 每一个小核的bn层
                branch_k, branch_b = fuse_bn_nondilated(conv, bn)     # 融合小核和对应的bn层
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)      # 把小核把空洞点置为0然后padding和大核一样大然后加到大核上，此处没有+号！！！
                origin_b += branch_b
            merged_conv = get_conv2d_dilated(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            print('dilated_branches have already reparamed')
            # 以下是删除属性，在处理输入张量时，如果试图访问这些属性将会跳过相关操作，所以在test中，直接就是out = self.origin_bn(self.lk_origin(x))
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


def get_dwconv(dim, kernel, bias, padding, dilation):
    return nn.Conv2d(dim, dim, kernel_size=kernel, stride=1, padding=padding, bias=bias, dilation=dilation, groups=dim)


class AMPooling_channel(nn.Module):
    def __init__(self, num_channels=256):
        super(AMPooling_channel, self).__init__()
        # self.avgpool = torch.max(attn, dim=1, keepdim=True)
        # self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        B, C, _, _ = x.size()
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_avg = torch.mean(x, dim=1, keepdim=True)
        avg_weights, max_weights = self.channel_attention(x_avg), self.channel_attention(x_max)
        band_weights = self.sigmoid(avg_weights + max_weights)
        return band_weights


class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)


class ASA(nn.Module):
    def __init__(self, dim, order=3):
        super().__init__()
        # self.groups = 1
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        # self.in_channels = sum(self.dims)
        self.dims.reverse()  # 根据 order 计算每一层的通道数。它们通过 dim 逐层减半，然后反转顺序
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        # self.identity = nn.BatchNorm2d(sum(self.dims))
        # self.dwconv1 = DilatedReparamBlock(sum(self.dims), 5, False)
        # self.dwconv1 = conv_bn(sum(self.dims), sum(self.dims), 3, 1, 1, 1, 1)
        self.dwconv1 = get_dwconv(sum(self.dims), 3, True, 1, 1)   # 应该改为LSK,sum(self.dims)=0.25+0.5+1
        # self.dwconv2 = get_dwconv(self.dims[1]+self.dims[2], 9, False, 4, 1)
        self.ReConv2 = DilatedReparamBlock(self.dims[1]+self.dims[2], 9, False)    # 这个DilatedReparamBlock也是深度卷积
        # self.dwconv3 = get_dwconv(self.dims[2], 29, False, 14, 1)
        self.ReConv3 = DilatedReparamBlock(self.dims[2], 19, False)
        self.AMPooling_channel1 = AMPooling_channel(sum(self.dims))
        self.AMPooling_channel2 = AMPooling_channel(sum(self.dims))
        self.AMPooling_channel3 = AMPooling_channel(sum(self.dims))
        self.proj_out = nn.Conv2d(dim, dim, 1)  # 图中的最后一层
        # self.interaction_out = nn.Conv2d(sum(self.dims), dim, 1)
        # self.pws是Mul上面的呢个调整通道的卷积层（入通道分别为0.25,0.5）
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        # in_channels = dim*3 // 2 ** 2
        # rate = 4
        # out_channels = in_channels
        # self.spatial_attention = nn.Sequential(
        #     nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
        #     nn.BatchNorm2d(int(in_channels / rate)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
        #     nn.BatchNorm2d(out_channels))
        # self.conv_squeeze = nn.Conv2d(3, 3, 7, padding=3)
        # self.reparam = False

    def forward(self, x, mask=None, dummy=False):
        # input = x
        # B, C, H, W = x.shape

        # fused_x = self.proj_in(x)
        # 将投影后的张量 fused_x 分割为两部分
        x = self.proj_in(x)
        pwa, abc = torch.split(x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv1(abc)+abc
        # if self.reparam:
        #     dw_abc = self.identity(abc)
        # else:
        #     dw_abc = self.dwconv1(abc)+self.identity(abc)   # A
        # abc, bcd, cde = torch.split(abc, (self.dims[0], self.dims[1], self.dims[2]), dim=1)
        bcd, cde = torch.split(dw_abc, (self.dims[0], self.dims[1]+self.dims[2]), dim=1)
        # dw_bcd = self.dwconv2(bcd)
        dw_cde = self.ReConv2(cde)   # B
        # 深度卷积的输出 dw_abc 分割成多部分，每部分的通道数分别为 self.dims 中的值(0.25,0.5,1)
        efg, fgh = torch.split(dw_cde, (self.dims[1], self.dims[2]), dim=1)
        dw_fgh = self.ReConv3(fgh)  # C
        attn1 = bcd
        # attn2 = self.conv_spatial(attn1)
        attn2 = efg
        attn3 = dw_fgh
        attn = torch.cat([attn1, attn2, attn3], dim=1)
        AMPooling_channel_attn1 = self.AMPooling_channel1(attn)
        AMPooling_channel_attn2 = self.AMPooling_channel2(attn)
        AMPooling_channel_attn3 = self.AMPooling_channel3(attn)

        # avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # soft_attn1 = channel_softpooling(attn)
        # soft_attn2 = channel_softpooling(attn)
        # soft_attn3 = channel_softpooling(attn)
        # max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # pooling层没有参数，直接加载参数test即可，但为何换不换pooling层计算出来的test结果是一样的，最后一句错，是不一样的！！！！
        # soft_attn = torch.cat([avg_attn, avg_attn, soft_attn3], dim=1)
        soft_attn = torch.cat([AMPooling_channel_attn1, AMPooling_channel_attn2, AMPooling_channel_attn3], dim=1)
        # soft_attn = self.spatial_attention(attn)
        # sig = self.conv_squeeze(soft_attn).sigmoid()
        attnbcd = attn1 * soft_attn[:, 0, :, :].unsqueeze(1)
        attnefg = attn2 * soft_attn[:, 1, :, :].unsqueeze(1)
        attnfgh = attn3 * soft_attn[:, 2, :, :].unsqueeze(1)
        dw_list = [attnbcd,
                   attnefg,
                   attnfgh]
        x = pwa * dw_list[0]
        # tensor_interaction = torch.cat([attnbcd, attnefg, attnfgh], dim=1)
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)
        # tensor_interaction = self.interaction_out(tensor_interaction)

        return x

    # def merge_kernel(self):
    #     kernelid, biasid = self._fuse_bn_tensor(self.identity)
    #     kernel3x3, bias3x3 = self._fuse_bn_tensor(self.dwconv1)
    #     k, b = kernel3x3 + kernelid, bias3x3 + biasid
    #     new_conv = nn.Conv2d(sum(self.dims), sum(self.dims), 3, 1, 1, 1)
    #     new_conv.weight.data = k
    #     new_conv.bias.data = b
    #     self.identity = new_conv
    #     self.reparam = True
    #
    # def _fuse_bn_tensor(self, branch):
    #     if branch is None:
    #         return 0, 0
    #     if isinstance(branch, nn.Sequential):
    #         kernel = branch.conv.weight
    #         running_mean = branch.bn.running_mean
    #         running_var = branch.bn.running_var
    #         gamma = branch.bn.weight
    #         beta = branch.bn.bias
    #         eps = branch.bn.eps
    #     else:
    #         assert isinstance(branch, nn.BatchNorm2d)
    #         if not hasattr(self, 'id_tensor'):
    #             input_dim = self.in_channels // self.groups
    #             kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
    #             for i in range(self.in_channels):
    #                 kernel_value[i, i % input_dim, 1, 1] = 1
    #             self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
    #         kernel = self.id_tensor
    #         running_mean = branch.running_mean
    #         running_var = branch.running_var
    #         gamma = branch.weight
    #         beta = branch.bias
    #         eps = branch.eps
    #     std = (running_var + eps).sqrt()
    #     t = (gamma / std).reshape(-1, 1, 1, 1)
    #     return kernel * t, beta - running_mean * gamma / std


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# class Block(nn.Module):
#     r""" HorNet block
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
#         super().__init__()
#
#         self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
#         self.gnconv = gnconv(dim) # depthwise conv
#         self.norm2 = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#
#         self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
#                                     requires_grad=True) if layer_scale_init_value > 0 else None
#
#         self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
#                                     requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W  = x.shape
#         if self.gamma1 is not None:
#             gamma1 = self.gamma1.view(C, 1, 1)
#         else:
#             gamma1 = 1
#         x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))
#
#         input = x
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)       # 调整形状是为了全连接层
#         x = self.norm2(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma2 is not None:
#             x = self.gamma2 * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         x = input + self.drop_path(x)
#         return x


def channel_shuffle(x, groups):
    # x[batch_size, channels, H, W]
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):

    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        if channels % groups != 0:
            raise ValueError("The number of channels must be divisible by the number of groups!")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)


class MFAEBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=0.5, drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]  # norm_cfg：规范化层的配置，用于构建规范化层。如果未提供，则使用默认的nn.BatchNorm2d
        else:
            self.norm1 = nn.BatchNorm2d(dim)
        self.attn = ALKC(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # nn.Identity()简单地将输入返回作为输出
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = PMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # 这才是真正的FFN
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)  # torch.ones((dim))这个操作将创建一个长度为 dim 的张量，其中的所有元素都是 1
        # self.c_shuffle = ChannelShuffle(dim, groups=4)
        self.relu = nn.ReLU(inplace=True)
        # self.HORBlock = Block(dim)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.norm1(self.attn(x)))  # 尽管只有三个维度，但形状为 [channels, 1, 1] 的张量会被扩展为 [1, channels, 1, 1]
        x = x + self.drop_path(self.mlp(x))
        out = self.relu(x)
        # out = self.c_shuffle(out)
        # x = self.HORBlock(x)
        return out


class SRModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.VRAttention = VRAttention(inplanes=d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        # shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x, att = self.VRAttention(x)
        x = self.proj_2(x)
        # x = x + shorcut
        return x, att


class GSRCBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]  # norm_cfg：规范化层的配置，用于构建规范化层。如果未提供，则使用默认的nn.BatchNorm2d
        else:
            self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SRModule(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # nn.Identity()简单地将输入返回作为输出
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # 这才是真正的FFN
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)  # torch.ones((dim))这个操作将创建一个长度为 dim 的张量，其中的所有元素都是 1

    def forward(self, x):
        y, att = self.attn(self.norm1(x))
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y)  # 尽管只有三个维度，但形状为 [channels, 1, 1] 的张量会被扩展为 [1, channels, 1, 1]
        x = x + self.drop_path(self.mlp(x))
        return x, att


class MsGsFNet(nn.Module):

    def __init__(self, input_nc, output_nc, patch_based: bool, inner_nc, n_blocks=3):
        assert (n_blocks >= 0)
        super(MsGsFNet, self).__init__()
        self.input_nc = input_nc
        filters = [inner_nc]
        # self.GSRCBlock = GSRCBlock(input_nc)
        self.patch_based = patch_based

        # self.MFAEBlock = MFAEBlock(filters[0])
        self.output_nc = output_nc

        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0], affine=True),
            # SE_Module(ngf * 1 * 2, ratio=16),
            nn.ReLU(True))
        if self.patch_based:
            self.lastconv = nn.Linear(filters[0], self.output_nc)
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        else:
            self.lastconv = nn.Conv2d(filters[0], self.output_nc, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            #
            # for i in range(n_blocks):
            #     model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]
            #     ResnetBlock的输出和输入一样，通道一样
            '''torch.Size([1, 32, 64, 64])
               torch.Size([1, 32, 64, 64])'''

    def forward(self, input):
        # x1, att = self.GSRCBlock(input)
        x1 = self.conv(input)  # 270->64

        # x1 = self.MFAEBlock(x1)
        if self.patch_based:
            x1 = self.avgpool(x1)
            x1 = x1.view(x1.size(0), -1)
            x1 = x1.detach()
            att.squeeze()

        output = self.lastconv(x1)
        return output, 1

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()
        for m in self.modules():
            if hasattr(m, 'merge_dilated_branches'):
                m.merge_dilated_branches()
