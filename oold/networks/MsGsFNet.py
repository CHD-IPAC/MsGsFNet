#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 06:00:09 2019

@author: aneesh
"""

import functools
import math
import torch.nn.functional as F
from functools import partial
from networks.VRAttention import VRAttention
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


# 注意卷积核的shape！！！！
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


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):   # dilated_r是空洞率，large_kernel是最大的核和bn层融合后的等效核，dilated_kernel小核的等效核
    large_k = large_kernel.size(2)   # 高度
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1       # 这就跟自己计算感受野的公式一样
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


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


class ASA(nn.Module):
    def __init__(self, dim, order=3):
        super().__init__()
        # self.groups = 1
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()  # 根据 order 计算每一层的通道数。它们通过 dim 逐层减半，然后反转顺序
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        self.dwconv1 = get_dwconv(sum(self.dims), 3, True, 1, 1)
        self.ReConv2 = DilatedReparamBlock(self.dims[1]+self.dims[2], 9, False)
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

    def forward(self, x):
        x = self.proj_in(x)
        pwa, abc = torch.split(x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv1(abc)+abc
        bcd, cde = torch.split(dw_abc, (self.dims[0], self.dims[1]+self.dims[2]), dim=1)
        dw_cde = self.ReConv2(cde)   # B
        # 深度卷积的输出 dw_abc 分割成多部分，每部分的通道数分别为 self.dims 中的值(0.25,0.5,1)
        efg, fgh = torch.split(dw_cde, (self.dims[1], self.dims[2]), dim=1)
        dw_fgh = self.ReConv3(fgh)  # C
        attn1 = bcd
        attn2 = efg
        attn3 = dw_fgh
        attn = torch.cat([attn1, attn2, attn3], dim=1)
        AMPooling_channel_attn1 = self.AMPooling_channel1(attn)
        AMPooling_channel_attn2 = self.AMPooling_channel2(attn)
        AMPooling_channel_attn3 = self.AMPooling_channel3(attn)
        soft_attn = torch.cat([AMPooling_channel_attn1, AMPooling_channel_attn2, AMPooling_channel_attn3], dim=1)
        attnbcd = attn1 * soft_attn[:, 0, :, :].unsqueeze(1)
        attnefg = attn2 * soft_attn[:, 1, :, :].unsqueeze(1)
        attnfgh = attn3 * soft_attn[:, 2, :, :].unsqueeze(1)
        dw_list = [attnbcd,
                   attnefg,
                   attnfgh]
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


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


class MFAEBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=0.5, drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]  # 使用默认的nn.BatchNorm2d
        else:
            self.norm1 = nn.BatchNorm2d(dim)
        self.attn = ALKC(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # nn.Identity()简单地将输入返回作为输出
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = PMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.norm1(self.attn(x)))
        x = x + self.drop_path(self.mlp(x))
        out = self.relu(x)
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
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]  # 使用默认的nn.BatchNorm2d
        else:
            self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SRModule(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # nn.Identity()简单地将输入返回作为输出
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)  # torch.ones((dim))这个操作将创建一个长度为 dim 的张量，其中的所有元素都是 1

    def forward(self, x):
        y, att = self.attn(self.norm1(x))
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y)
        x = x + self.drop_path(self.mlp(x))
        return x, att


class MsGsFNet(nn.Module):

    def __init__(self, input_nc, output_nc, patch_based: bool, inner_nc, n_blocks=3):
        assert (n_blocks >= 0)
        super(MsGsFNet, self).__init__()
        self.input_nc = input_nc
        filters = [inner_nc]
        self.GSRCBlock = GSRCBlock(input_nc)
        self.patch_based = patch_based

        self.MFAEBlock = MFAEBlock(filters[0])
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

    def forward(self, input):
        x1, att = self.GSRCBlock(input)
        x1 = self.conv(x1)  # 270->64

        x1 = self.MFAEBlock(x1)
        if self.patch_based:
            x1 = self.avgpool(x1)
            x1 = x1.view(x1.size(0), -1)
            x1 = x1.detach()
            att.squeeze()

        output = self.lastconv(x1)
        return output, att

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()
        for m in self.modules():
            if hasattr(m, 'merge_dilated_branches'):
                m.merge_dilated_branches()
