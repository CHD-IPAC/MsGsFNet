#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:11:14 2019

@author: aneesh
"""

import os
import os.path as osp
from os import listdir

import imageio
import torch
import torch.utils.data as data
import cv2
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix
from os.path import splitext


def tensor_to_image(torch_tensor, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)):
    '''
    Converts a 3D Pytorch tensor into a numpy array for display
    
    Parameters:
        torch_tensor -- Pytorch tensor in format(channels, height, width)
    '''
    for t, m, s in zip(torch_tensor, mean, std):
        t.mul_(s).add_(m)

    return np.uint8(torch_tensor.mul(255.0).numpy().transpose(1, 2, 0))

class AverageMeter(object):               # 初始就reset
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metrics():
    '''
    Calculates all the metrics reported in paper: Overall Accuracy, Average Accuracy,
    mean IOU and mean DICE score
    Ref: https://github.com/rmkemker/EarthMapper/blob/master/metrics.py
    
    Parameters:
        ignore_index -- which particular index to ignore when calculating all values.
                        In AeroRIT, index '5' is the undefined class and hence, the 
                        default value for this function.
    '''
    def __init__(self, ignore_index=0):
        self.ignore_index = ignore_index

    def __call__(self, truth, prediction):            # len(prediction) = 123904

        ignore_locs = np.where(truth == self.ignore_index)
        # print(ignore_locs)
        # print(len(ignore_locs[0]))
        truth = np.delete(truth, ignore_locs)            # 这是为了计算，直接把背景信息discard
        print('hunxiaozhiqian max index',np.max(truth))  # 看现在抛弃背景信息后是有几个类别，这里输出的是最大的类别号 此类别号+1就是现在的类别数，+2就是原来总的num_classes
        # print('ignore_locs:',len(ignore_locs[0]))#66003
        # print('ignore_locs2.shape:',len(ignore_locs[1]))
        # print('prediction:',prediction.shape)
        # print('truth:',truth.shape)

        prediction = np.delete(prediction, ignore_locs)     # 216064-66003=150061      会根据 ignore_locs 数组中的索引值，删除 prediction 数组中对应位置的元素
        # print('prediction after delate:',prediction.shape)
        # 什么是混淆矩阵：
        # [[2 0 0]
        #  [0 0 1]
        #  [1 0 2]]这里，行表示真实类别，列表示预测类别，元素是把某一类预测为某一类的个数。以第一行为例：2表示模型正确预测了2个类别为0的样本，两个0分别表示把0个类别0预测了成了1，2。

        # 生成混淆矩阵。生成混淆矩阵之前，truth和prediction都是被展平了的。
        self.c = confusion_matrix(truth, prediction)
        hun = self.c
        # print('second hunxiaojuzhen shape', hun.shape)      # 混淆矩阵的shape一定是(类别数，类别数)
        # print('second hunxiaojuzhen:\n',hun)
        # return self._oa(), self._aa(), self._mIOU(), self._dice_coefficient(), self._IOU()
        return self._oa(), self._aa(), self._mIOU(), self._dice_coefficient(), self._IOU(), self._f1Score(), self._precision(), self._recall()

    def _oa(self):
        # 所有正确预测除以所有样本数，也可以说是所有正确预测除以预测的次数
        # np.diag(self.c)就是正确预测的数量，np.sum(self.c)是所有预测情况总和也就是所有样本（像素点）的总数
        return np.sum(np.diag(self.c))/np.sum(self.c)

    def _f1Score(self):
        np.seterr(divide="ignore", invalid="ignore")
        precision = np.diag(self.c) / np.sum(self.c, axis=0)    # np.sum(self.c, axis=0)是每一列的总和，表示预测为这些类的总数。结果为每个类别的精确率，即（2/3,0/0,2/3）。此精确率表示的是预测的有效率。
        recall = np.diag(self.c) / np.sum(self.c, axis=1)       # np.sum(self.c, axis=1)表示每个真实类别的样本总数。得到每个类别的召回率
        f1score = 2 * precision * recall / (precision + recall)      # F1 分数是精确率和召回率的调和平均数
        return f1score
        # return np.nanmean(f1score)

    def _precision(self):                                   # 此精确率表示的是预测为此类别的有效率。预测对的个数除以预测为此类别的个数。
        precision = np.diag(self.c)/np.sum(self.c, axis=0)
        return precision
        # return np.nanmean(precision)
    def _recall(self):                                      # 此类别预测对的数量除以此类别真实数量。
        np.seterr(divide="ignore", invalid="ignore")
        temp = np.diag(self.c)/np.sum(self.c, axis=1)
        return temp
        # return np.nanmean(temp)

    def _aa(self):                                         # 即recall的平均值版本
        return np.nanmean(np.diag(self.c)/(np.sum(self.c, axis=1) + 1e-10))  # np.nanmean计算数组中非 NaN 值的平均值

    def _IOU(self):
        intersection = np.diag(self.c)    # 交集
        ground_truth_set = self.c.sum(axis=1)  # 此类别的总数
        predicted_set = self.c.sum(axis=0)     # 预测为此类别的总数
        union =  ground_truth_set + predicted_set - intersection + 1e-10         # 二者相加减去重合部分，即并集。

        intersection_over_union = intersection / union.astype(np.float32)        # 交并比
        return intersection_over_union

    def _mIOU(self):   # 交并比的平均值
        intersection_over_union = self._IOU()
        return np.nanmean(intersection_over_union)

    def _dice_coefficient(self):   # Dice 系数
        intersection = np.diag(self.c)
        ground_truth_set = self.c.sum(axis=1)
        predicted_set = self.c.sum(axis=0)
        dice = (2 * intersection) / (ground_truth_set + predicted_set + 1e-10)
        avg_dice = np.nanmean(dice)
        return avg_dice

    def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
        np.seterr(divide='ignore', invalid='ignore')
        freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  # 每个类别在整个数据集中的频率
        iu = np.diag(confusionMatrix) / (  # 交并比
                np.sum(confusionMatrix, axis=1) +
                np.sum(confusionMatrix, axis=0) -
                np.diag(confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()  # 计算每个类别的频率加权 IoU，并求和
        return FWIoU

class AeroCLoader(data.Dataset):
    '''
    This function serves as the dataloader for the AeroCampus dataset
    
    Parameters:
        set_loc     -- 'left', 'mid' or 'right' -> indicates which set is to be used
        set_type    -- 'train' or 'test'
        size        -- default 'small' -> 64 x 64.
        hsi_sign    -- 'rad' or 'ref' -> based on which hyperspectral image type is used
        hsi_mode    -- available sampling options are:
                        '3b' -> samples 7, 15, 25 (BGR)
                        '4b' -> samples bands 7, 15, 25, 46 (BGR + IR)
                        '6b' -> samples bands 7, 15, 25, 33, 40, 50 (BGR + 3x IR)
                        'visible' -> samples all visible bands
                        'all' -> samples all 51 bands (visible + infrared)
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    '''
# valset = AeroCLoader(set_loc='mid', set_type='test', size='small',hsi_sign=args.hsi_c, hsi_mode=hsi_mode, transforms=tx)
    def __init__(self, set_loc = 'left', set_type = 'train', size = 'small', hsi_sign = 'rad',
                 hsi_mode = 'all', transforms = None, augs = None):

        if size == 'small':
            size = '64'
        else:
            raise Exception('Size not present in the dataset')

        self.working_dir = 'Image' + size
        self.working_dir = osp.join('Aerial Data', self.working_dir, 'Data-' + set_loc)  # 保证Aerial Data文件夹在此项目之下

        # self.rgb_dir = 'RGB'
        self.label_dir = 'Labels'
        self.hsi_sign = hsi_sign
        self.hsi_dir = 'HSI' + '-{}'.format(self.hsi_sign)

        self.transforms = transforms
        self.augmentations = augs

        self.hsi_mode = hsi_mode
        self.hsi_dict = {
                '3b':[7, 15, 25],
                '4b':[7, 15, 25, 46],
                '6b':[7, 15, 25, 33, 40, 50],
                'visible':'all 400 - 700 nm',
                'all': 'all 51 bands'}

        self.n_classes = len(self.get_labels())

        with open(osp.join(self.working_dir, set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()

    def __getitem__(self, index):
        # rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.tif'))
        # rgb = rgb[:,:,::-1]

        hsi = np.load(osp.join(self.working_dir, self.hsi_dir, self.filelist[index] + '.npy'))
        idx = self.filelist[index]

        if self.hsi_mode == 'visible':
            hsi = hsi[:,:,0:31]
        elif self.hsi_mode == 'all':
            hsi = hsi
        else:
            bands = self.hsi_dict[self.hsi_mode]
            hsi_temp = np.zeros((hsi.shape[0], hsi.shape[1], len(bands)))
            for i in range(len(bands)):
                hsi_temp[:,:,i] = hsi[:,:,bands[i]]
            hsi = hsi_temp

        hsi = hsi.astype(np.float32)

        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = label[:,:,::-1]   # 会将每个像素的通道顺序反转

        if self.augmentations is not None:
            # rgb, hsi, label = self.augmentations(rgb, hsi, label)
            hsi, label = self.augmentations(hsi, label)

        if self.transforms is not None:  # 并没有用主文件中定义的tx来预处理图像
            # rgb = self.transforms(rgb)

            if self.hsi_sign == 'rad':
                hsi = np.clip(hsi, 0, 2**14)/2**14    # 这是论文中的归一化方法，最大值是2的14次方。其实也可以标准化
                hsi = np.transpose(hsi, (2, 0, 1))    # HWC（Height, Width, Channels）：高、宽、通道，这通常是图像存储的原始格式。
                                                      # CHW（Channels, Height, Width）：通道、高、宽，这通常是深度学习框架（如 PyTorch）中处理图像数据的格式。
                hsi = torch.from_numpy(hsi)
            elif self.hsi_sign == 'ref':
                hsi = np.clip(hsi, 0, 100)/100  # max=691.2094,min=-11.4483
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)

            label = self.encode_segmap(label)
            label = torch.from_numpy(np.array(label)).long()

        return {'hsi': hsi,
                'label': label,
                'idx': idx}

    def __len__(self):
        return len(self.filelist)

    def get_labels(self):
        return np.asarray(
                [
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                        [0, 255, 255],
                        [255, 127, 80],
                        [153, 0, 0],
                        ]
                )

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return np.uint8(rgb)

class XiongAnLoader(data.Dataset):
    '''
    This function serves as the dataloader for the AeroCampus dataset

    Parameters:
        set_loc     -- 'left', 'mid' or 'right' -> indicates which set is to be used
        set_type    -- 'train' or 'test'
        size        -- default 'small' -> 64 x 64.
        hsi_sign    -- 'rad' or 'ref' -> based on which hyperspectral image type is used
        hsi_mode    -- available sampling options are:
                        '3b' -> samples 7, 15, 25 (BGR)
                        '4b' -> samples bands 7, 15, 25, 46 (BGR + IR)
                        '6b' -> samples bands 7, 15, 25, 33, 40, 50 (BGR + 3x IR)
                        'visible' -> samples all visible bands
                        'all' -> samples all 51 bands (visible + infrared)
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    '''

    def __init__(self, set_type='train', size='small', hsi_sign='rad',
                 hsi_mode='all', transforms=None, augs=None):
        self.set_type = set_type

        if size == 'small':
            size = '64'
        else:
            raise Exception('Size not present in the dataset')
        # if set_loc == 'right' and set_type == 'test':
        #     self.working_dir = 'Image' + size + '_no_overlap'  # no_overlap 810 used for test
        # else:
        # self.working_dir = 'Image' + size #overlap 3127
        # self.working_dir = osp.join('XiongAn', self.working_dir, 'Data-' + set_loc)
        self.working_dir = 'Image' + size +'_step_patch'#overlap 3127
        self.working_dir = osp.join('XiongAn', self.working_dir)

        # self.rgb_dir = 'RGB'

        # self.label_dir = 'Labels'
        self.label_dir = 'Labels_no_17'
        # self.specific_label_dir = osp.join(self.working_dir, self.label_dir)
        self.hsi_sign = hsi_sign
        self.hsi_dir = 'HSI' + '-{}'.format(self.hsi_sign)

        self.transforms = transforms
        self.augmentations = augs

        self.hsi_mode = hsi_mode
        self.hsi_dict = {
            '3b': [7, 15, 25],
            '4b': [7, 15, 25, 46],
            '6b': [7, 15, 25, 33, 40, 50],
            'visible': 'all 400 - 700 nm',
            'all': 'all 51 bands'}

        self.n_classes = len(self.get_labels()) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        with open(osp.join(self.working_dir, set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()
        # self.ids = [splitext(file)[0] for file in listdir(self.specific_label_dir)
        #             if not file.startswith('.')]

    def __getitem__(self, index):
        # idx = self.ids[index]
        # rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.tif'))
        # rgb = rgb[:, :, ::-1]

        hsi = np.load(osp.join(self.working_dir, self.hsi_dir, self.filelist[index] + '.npy'))
        # print("hsi:",hsi.shape)
        idx = self.filelist[index]
        # print('idx:',self.set_type,idx)
        if self.hsi_mode == 'visible':
            hsi = hsi[:, :, 0:31]
        elif self.hsi_mode == 'all':
            hsi = hsi
        else:
            bands = self.hsi_dict[self.hsi_mode]
            hsi_temp = np.zeros((hsi.shape[0], hsi.shape[1], len(bands)))
            for i in range(len(bands)):
                hsi_temp[:, :, i] = hsi[:, :, bands[i]]
            hsi = hsi_temp

        hsi = hsi.astype(np.float32)

        # label = imageio.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        # print('label',label.shape)
        label = label[:, :, ::-1]
        # print('-1 zhihou type', type(label))#<class 'numpy.ndarray'>
        # print('-1 zhihou', label.shape)#!!!!!!!!!!!!!!!!!!!!!!!1 (64, 64, 3)

        if self.augmentations is not None:
            hsi, label = self.augmentations(hsi, label)
            # print('augmentations type', type(label))#<class 'numpy.ndarray'>
            # print('augmentations', label.shape)# !!!!!!!!!!!!!!!!(64, 64, 3)

        if self.transforms is not None:
            # rgb = self.transforms(rgb)

            if self.hsi_sign == 'rad':
                # hsi = np.clip(hsi, 0, 2 ** 14) / 2 ** 14
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)  # np转为tensor
            elif self.hsi_sign == 'ref':
                # hsi = np.clip(hsi, 0, 100) / 100
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            label = self.encode_segmap(label)
            # print('encode_segmap type', type(label))#                                 <class 'numpy.ndarray'>
            # print('encode_segmap', label.shape)# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!(64, 64)
            label = torch.from_numpy(np.array(label)).long()
            # print('from_numpy type', type(label))#   <class 'torch.Tensor'>
            # print('from_numpy', label.shape)#        torch.Size([64, 64])
            # print('dtype',label.dtype)
            # print(label)
        # idex_category = self.get_idex_by_category(idx, label)

        # return hsi, label
        return {'hsi': hsi,
                'label': label,
                'idx': idx}

    def __len__(self):
        return len(self.filelist)

    # def get_idex_by_category(self, idx, label):
    #     label = label.numpy()
    #     if not(np.any(label)):
    #
    #             f.write("%s\n" % idx)
            # print()
        # return index

    def get_labels(self):
        return np.asarray(
            [
                [0,   0,   0],
                [0, 139,   0],
                [0,   0, 255],
                [255, 255, 0],# 3
                [0, 255, 255],#4
                [255, 0, 255],
                [139, 139, 0],# 6
                [0, 139, 139],
                [0, 255,   0],#
                [0,   0, 139],#9
                [255, 127,80],
                [127, 255, 0],#11-5612 2.62%
                [218,112,214],
                [46, 139, 87],
                [131 ,111, 255],#14-7151
                [255, 165, 0],
                [127,255,212],
                # [196, 196, 0],#17-1496系柳林原本为：(218,112,214)和玉米颜色重了
                [255,  0,  0],
                [205,  0,  0],
                [139, 0, 0],
            ]
        )

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return np.uint8(rgb)

class LongKouLoader(data.Dataset):
    '''
    This function serves as the dataloader for the AeroCampus dataset

    Parameters:
        set_loc     -- 'left', 'mid' or 'right' -> indicates which set is to be used
        set_type    -- 'train' or 'test'
        size        -- default 'small' -> 64 x 64.
        hsi_sign    -- 'rad' or 'ref' -> based on which hyperspectral image type is used
        hsi_mode    -- available sampling options are:
                        '3b' -> samples 7, 15, 25 (BGR)
                        '4b' -> samples bands 7, 15, 25, 46 (BGR + IR)
                        '6b' -> samples bands 7, 15, 25, 33, 40, 50 (BGR + 3x IR)
                        'visible' -> samples all visible bands
                        'all' -> samples all 51 bands (visible + infrared)
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    '''

    def __init__(self, set_type='train', size='small', hsi_sign='rad',
                 hsi_mode='all', transforms=None, augs=None):
        self.set_type = set_type

        if size == 'small':
            size = '32'
        else:
            raise Exception('Size not present in the dataset')
        # if set_loc == 'right' and set_type == 'test':
        #     self.working_dir = 'Image' + size + '_no_overlap'  # no_overlap 810 used for test
        # else:
        # self.working_dir = 'Image' + size #overlap 3127
        # self.working_dir = osp.join('XiongAn', self.working_dir, 'Data-' + set_loc)
        self.working_dir = 'Image' + size +'_step_patch'#overlap 3127
        self.working_dir = osp.join('WHU-Hi-LongKou', self.working_dir)

        # self.rgb_dir = 'RGB'
        self.label_dir = 'Labels'
        self.specific_label_dir = osp.join(self.working_dir, self.label_dir)
        self.hsi_sign = hsi_sign
        self.hsi_dir = 'HSI' + '-{}'.format(self.hsi_sign)

        self.transforms = transforms
        self.augmentations = augs

        self.hsi_mode = hsi_mode
        self.hsi_dict = {
            '3b': [7, 15, 25],
            '4b': [7, 15, 25, 46],
            '6b': [7, 15, 25, 33, 40, 50],
            'visible': 'all 400 - 700 nm',
            'all': 'all 51 bands'}

        self.n_classes = len(self.get_labels()) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!，len是输出行数，为10

        with open(osp.join(self.working_dir, set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()                               # 一个个图片的name
        # self.ids = [splitext(file)[0] for file in listdir(self.specific_label_dir)
        #             if not file.startswith('.')]

    def __getitem__(self, index):                        # 在创建了一个LongKouLoader对象之后，eg主函数中：当你使用 enumerate(trainloader, 0) 对 trainloader 进行迭代时，
                                                         # 会自动调用 trainloader 对象的 __getitem__ 方法来获取每个批次的数据，迭代时index 会从 0 开始递增
        # idx = self.ids[index]
        # rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.tif'))
        # rgb = rgb[:, :, ::-1]

        hsi = np.load(osp.join(self.working_dir, self.hsi_dir, self.filelist[index] + '.npy'))
        idx = self.filelist[index]
        # print('idx:',self.set_type,idx)
        if self.hsi_mode == 'visible':
            hsi = hsi[:, :, 0:31]
        elif self.hsi_mode == 'all':
            hsi = hsi
        else:
            bands = self.hsi_dict[self.hsi_mode]
            hsi_temp = np.zeros((hsi.shape[0], hsi.shape[1], len(bands)))
            for i in range(len(bands)):
                hsi_temp[:, :, i] = hsi[:, :, bands[i]]
            hsi = hsi_temp

        hsi = hsi.astype(np.float32)

        # label = imageio.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = label[:, :, ::-1]         # 从数组末尾开始逆序选择元素，这是因为OpenCV中，图像读取后的通道顺序是BGR
        # print('-1 zhihou type', type(label))#<class 'numpy.ndarray'>
        # print('-1 zhihou', label.shape)#!!!!!!!!!!!!!!!!!!!!!!!1 (64, 64, 3)

        if self.augmentations is not None:
            hsi, label = self.augmentations(hsi, label)
            # print('augmentations type', type(label))#<class 'numpy.ndarray'>
            # print('augmentations', label.shape)# !!!!!!!!!!!!!!!!(64, 64, 3)

        if self.transforms is not None:
            # rgb = self.transforms(rgb)

            if self.hsi_sign == 'rad':
                # hsi = np.clip(hsi, 0, 2 ** 14) / 2 ** 14
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)  # np转为tensor
            elif self.hsi_sign == 'ref':
                # hsi = np.clip(hsi, 0, 100) / 100
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            label = self.encode_segmap(label)
            # print('encode_segmap type', type(label))#                                 <class 'numpy.ndarray'>
            # print('encode_segmap', label.shape)# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!(64, 64)
            label = torch.from_numpy(np.array(label)).long()
            # print('from_numpy type', type(label))#   <class 'torch.Tensor'>
            # print('from_numpy', label.shape)#        torch.Size([64, 64])
            # print('dtype',label.dtype)
            # print(label)
        # idex_category = self.get_idex_by_category(idx, label)

        # return hsi, label
        return {'hsi': hsi,
                'label': label,
                'idx': idx}

    def __len__(self):
        return len(self.filelist)

    # def get_idex_by_category(self, idx, label):
    #     label = label.numpy()
    #     if not(np.any(label)):
    #
    #             f.write("%s\n" % idx)
            # print()
        # return index

    def get_labels(self):
        return np.asarray(
            [
                [0, 0, 0],
                [0, 139, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],  # 5
                [255, 0, 255],
                [139, 139, 0],
                [0, 139, 139],
                [0, 255, 0],
                [0, 0, 139],  # 10
            ]
        )

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return np.uint8(rgb)

class HongHuLoader(data.Dataset):
    '''
    This function serves as the dataloader for the AeroCampus dataset

    Parameters:
        set_loc     -- 'left', 'mid' or 'right' -> indicates which set is to be used
        set_type    -- 'train' or 'test'
        size        -- default 'small' -> 64 x 64.
        hsi_sign    -- 'rad' or 'ref' -> based on which hyperspectral image type is used
        hsi_mode    -- available sampling options are:
                        '3b' -> samples 7, 15, 25 (BGR)
                        '4b' -> samples bands 7, 15, 25, 46 (BGR + IR)
                        '6b' -> samples bands 7, 15, 25, 33, 40, 50 (BGR + 3x IR)
                        'visible' -> samples all visible bands
                        'all' -> samples all 51 bands (visible + infrared)
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    '''

    def __init__(self, set_type='train', size='small', hsi_sign='rad',
                 hsi_mode='all', transforms=None, augs=None):
        self.set_type = set_type

        if size == 'small':
            size = '32'
        else:
            raise Exception('Size not present in the dataset')
        # if set_loc == 'right' and set_type == 'test':
        #     self.working_dir = 'Image' + size + '_no_overlap'  # no_overlap 810 used for test
        # else:
        # self.working_dir = 'Image' + size #overlap 3127
        # self.working_dir = osp.join('XiongAn', self.working_dir, 'Data-' + set_loc)
        self.working_dir = 'Image' + size +'_step_patch'#overlap 3127
        self.working_dir = osp.join('WHU-Hi-HongHu', self.working_dir)

        # self.rgb_dir = 'RGB'
        self.label_dir = 'Labels'
        # self.specific_label_dir = osp.join(self.working_dir, self.label_dir)
        self.hsi_sign = hsi_sign
        self.hsi_dir = 'HSI' + '-{}'.format(self.hsi_sign)

        self.transforms = transforms
        self.augmentations = augs

        self.hsi_mode = hsi_mode
        self.hsi_dict = {
            '3b': [7, 15, 25],
            '4b': [7, 15, 25, 46],
            '6b': [7, 15, 25, 33, 40, 50],
            'visible': 'all 400 - 700 nm',
            'all': 'all 51 bands'}

        self.n_classes = len(self.get_labels()) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        with open(osp.join(self.working_dir, set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()            # 将f的每一行变成列表中的每一个元素
        # self.ids = [splitext(file)[0] for file in listdir(self.specific_label_dir)
        #             if not file.startswith('.')]

    def __getitem__(self, index):                        # 在创建了一个LongKouLoader对象之后，eg主函数中：当你使用 enumerate(trainloader, 0) 对 trainloader 进行迭代时，
                                                         # 会自动调用 trainloader 对象的 __getitem__ 方法来获取每个批次的数据，迭代时index 会从 0 开始递增
                                                         # index 参数的最大值应该等于数据集的大小减去1即index 的最大值为 len(self.filelist) - 1。
        # idx = self.ids[index]
        # rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.tif'))
        # rgb = rgb[:, :, ::-1]

        # hsi = np.load(osp.join(self.working_dir, self.hsi_dir, self.filelist[index] + '.npy'))
        hsi = np.load(osp.join(self.working_dir, self.hsi_dir, 'image_800_288' + '.npy'))
        idx = self.filelist[index]
        # print('idx:',self.set_type,idx)
        if self.hsi_mode == 'visible':
            hsi = hsi[:, :, 0:31]
        elif self.hsi_mode == 'all':
            hsi = hsi
        else:
            bands = self.hsi_dict[self.hsi_mode]
            hsi_temp = np.zeros((hsi.shape[0], hsi.shape[1], len(bands)))
            for i in range(len(bands)):
                hsi_temp[:, :, i] = hsi[:, :, bands[i]]
            hsi = hsi_temp

        hsi = hsi.astype(np.float32)

        # label = imageio.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = label[:, :, ::-1]
        # print('-1 zhihou type', type(label))#<class 'numpy.ndarray'>
        # print('-1 zhihou', label.shape)#!!!!!!!!!!!!!!!!!!!!!!!1 (64, 64, 3)

        if self.augmentations is not None:
            hsi, label = self.augmentations(hsi, label)
            # print('augmentations type', type(label))#<class 'numpy.ndarray'>
            # print('augmentations', label.shape)# !!!!!!!!!!!!!!!!(64, 64, 3)

        if self.transforms is not None:
            # rgb = self.transforms(rgb)

            if self.hsi_sign == 'rad':
                # hsi = np.clip(hsi, 0, 2 ** 14) / 2 ** 14
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)  # np转为tensor
            elif self.hsi_sign == 'ref':
                # hsi = np.clip(hsi, 0, 100) / 100
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            label = self.encode_segmap(label)
            # print('encode_segmap type', type(label))#                                 <class 'numpy.ndarray'>
            # print('encode_segmap', label.shape)# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!(64, 64)
            label = torch.from_numpy(np.array(label)).long()
            # print('from_numpy type', type(label))#   <class 'torch.Tensor'>
            # print('from_numpy', label.shape)#        torch.Size([64, 64])
            # print('dtype',label.dtype)
            # print(label)
        # idex_category = self.get_idex_by_category(idx, label)

        # return hsi, label
        return {'hsi': hsi,
                'label': label,
                'idx': idx}

    def __len__(self):
        return len(self.filelist)

    # def get_idex_by_category(self, idx, label):
    #     label = label.numpy()
    #     if not(np.any(label)):
    #
    #             f.write("%s\n" % idx)
            # print()
        # return index

    def get_labels(self):
        return np.asarray(
            [
                [0,   0,   0],# 0
                [0, 139,   0],# 1
                [0,   0, 255],
                [255, 255, 0],
                [0, 255, 255],#4
                [255, 0, 255],
                [139, 139, 0],
                [0, 139, 139],
                [0, 255,   0],
                [0,   0, 139],#19
                [255, 127,80],
                [127, 255, 0],
                [218,112,214],
                [46, 139, 87],
                [131 ,111, 255],#14
                [255, 165, 0],
                [127,255,212],
                [196, 196, 0],
                [255,  0,  0],
                [205,  0,  0],#19
                [139,  0,  0],
                [255, 205, 0],
                [139, 205, 139]#22
            ]
        )

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_labels()):
            # print('dayin ii',ii,label)
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return np.uint8(rgb)

class HanChuanLoader(data.Dataset):
    '''
    This function serves as the dataloader for the AeroCampus dataset

    Parameters:
        set_loc     -- 'left', 'mid' or 'right' -> indicates which set is to be used
        set_type    -- 'train' or 'test'
        size        -- default 'small' -> 64 x 64.
        hsi_sign    -- 'rad' or 'ref' -> based on which hyperspectral image type is used
        hsi_mode    -- available sampling options are:
                        '3b' -> samples 7, 15, 25 (BGR)
                        '4b' -> samples bands 7, 15, 25, 46 (BGR + IR)
                        '6b' -> samples bands 7, 15, 25, 33, 40, 50 (BGR + 3x IR)
                        'visible' -> samples all visible bands
                        'all' -> samples all 51 bands (visible + infrared)
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    '''

    def __init__(self, set_type='train', size='small', hsi_sign='rad',
                 hsi_mode='all', transforms=None, augs=None):
        self.set_type = set_type

        if size == 'small':
            size = '32'
        else:
            raise Exception('Size not present in the dataset')
        # if set_loc == 'right' and set_type == 'test':
        #     self.working_dir = 'Image' + size + '_no_overlap'  # no_overlap 810 used for test
        # else:
        # self.working_dir = 'Image' + size #overlap 3127
        # self.working_dir = osp.join('XiongAn', self.working_dir, 'Data-' + set_loc)
        self.working_dir = 'Image' + size +'_step_patch'#overlap 3127
        self.working_dir = osp.join('WHU-Hi-HanChuan', self.working_dir)

        # self.rgb_dir = 'RGB'
        self.label_dir = 'Labels'
        self.specific_label_dir = osp.join(self.working_dir, self.label_dir)
        self.hsi_sign = hsi_sign
        self.hsi_dir = 'HSI' + '-{}'.format(self.hsi_sign)

        self.transforms = transforms
        self.augmentations = augs

        self.hsi_mode = hsi_mode
        self.hsi_dict = {
            '3b': [7, 15, 25],
            '4b': [7, 15, 25, 46],
            '6b': [7, 15, 25, 33, 40, 50],
            'visible': 'all 400 - 700 nm',
            'all': 'all 51 bands'}

        self.n_classes = len(self.get_labels()) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        with open(osp.join(self.working_dir, set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()
        # self.ids = [splitext(file)[0] for file in listdir(self.specific_label_dir)
        #             if not file.startswith('.')]

    def __getitem__(self, index):
        # idx = self.ids[index]
        # rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.tif'))
        # rgb = rgb[:, :, ::-1]

        hsi = np.load(osp.join(self.working_dir, self.hsi_dir, self.filelist[index] + '.npy'))
        idx = self.filelist[index]
        # print('idx:',self.set_type,idx)
        if self.hsi_mode == 'visible':
            hsi = hsi[:, :, 0:31]
        elif self.hsi_mode == 'all':
            hsi = hsi
        else:
            bands = self.hsi_dict[self.hsi_mode]
            hsi_temp = np.zeros((hsi.shape[0], hsi.shape[1], len(bands)))
            for i in range(len(bands)):
                hsi_temp[:, :, i] = hsi[:, :, bands[i]]
            hsi = hsi_temp

        hsi = hsi.astype(np.float32)

        # label = imageio.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = label[:, :, ::-1]
        # print('-1 zhihou type', type(label))#<class 'numpy.ndarray'>
        # print('-1 zhihou', label.shape)#!!!!!!!!!!!!!!!!!!!!!!!1 (64, 64, 3)

        if self.augmentations is not None:
            hsi, label = self.augmentations(hsi, label)
            # print('augmentations type', type(label))#<class 'numpy.ndarray'>
            # print('augmentations', label.shape)# !!!!!!!!!!!!!!!!(64, 64, 3)

        if self.transforms is not None:
            # rgb = self.transforms(rgb)

            if self.hsi_sign == 'rad':
                # hsi = np.clip(hsi, 0, 2 ** 14) / 2 ** 14
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)  # np转为tensor
            elif self.hsi_sign == 'ref':
                # hsi = np.clip(hsi, 0, 100) / 100
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            label = self.encode_segmap(label)
            # print('encode_segmap type', type(label))#                                 <class 'numpy.ndarray'>
            # print('encode_segmap', label.shape)# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!(64, 64)
            label = torch.from_numpy(np.array(label)).long()
            # print('from_numpy type', type(label))#   <class 'torch.Tensor'>
            # print('from_numpy', label.shape)#        torch.Size([64, 64])
            # print('dtype',label.dtype)
            # print(label)
        # idex_category = self.get_idex_by_category(idx, label)

        # return hsi, label
        return {'hsi': hsi,
                'label': label,
                'idx': idx}

    def __len__(self):
        return len(self.filelist)

    # def get_idex_by_category(self, idx, label):
    #     label = label.numpy()
    #     if not(np.any(label)):
    #
    #             f.write("%s\n" % idx)
            # print()
        # return index

    def get_labels(self):
        return np.asarray(
            [
                [0, 0,   0], # 0
                [0, 139, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],  # 4
                [255, 0, 255],
                [139, 139, 0],
                [0, 139, 139],
                [0, 255, 0],
                [0, 0, 139],  # 9
                [255, 127, 80],
                [127, 255, 0],
                [218, 112, 214],
                [46, 139, 87],
                [131 ,111, 255],#14
                [255, 165, 0],
                [127, 255, 212],
            ]
        )

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return np.uint8(rgb)

class ShandongDowntownLoader(data.Dataset):
    '''
    This function serves as the dataloader for the AeroCampus dataset

    Parameters:
        set_loc     -- 'left', 'mid' or 'right' -> indicates which set is to be used
        set_type    -- 'train' or 'test'
        size        -- default 'small' -> 64 x 64.
        hsi_sign    -- 'rad' or 'ref' -> based on which hyperspectral image type is used
        hsi_mode    -- available sampling options are:
                        '3b' -> samples 7, 15, 25 (BGR)
                        '4b' -> samples bands 7, 15, 25, 46 (BGR + IR)
                        '6b' -> samples bands 7, 15, 25, 33, 40, 50 (BGR + 3x IR)
                        'visible' -> samples all visible bands
                        'all' -> samples all 51 bands (visible + infrared)
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    '''

    def __init__(self, set_type='train', size='small', hsi_sign='rad',
                 hsi_mode='all', transforms=None, augs=None):
        self.set_type = set_type

        if size == 'small':
            size = '64'
        else:
            raise Exception('Size not present in the dataset')
        # if set_loc == 'right' and set_type == 'test':
        #     self.working_dir = 'Image' + size + '_no_overlap'  # no_overlap 810 used for test
        # else:
        # self.working_dir = 'Image' + size #overlap 3127
        # self.working_dir = osp.join('XiongAn', self.working_dir, 'Data-' + set_loc)
        self.working_dir = 'Image' + size +'_step_patch'#overlap 3127
        self.working_dir = osp.join('ShandongDowntown', self.working_dir)

        # self.rgb_dir = 'RGB'
        self.label_dir = 'Labels'
        self.specific_label_dir = osp.join(self.working_dir, self.label_dir)
        self.hsi_sign = hsi_sign
        self.hsi_dir = 'HSI' + '-{}'.format(self.hsi_sign)

        self.transforms = transforms
        self.augmentations = augs

        self.hsi_mode = hsi_mode
        self.hsi_dict = {
            '3b': [7, 15, 25],
            '4b': [7, 15, 25, 46],
            '6b': [7, 15, 25, 33, 40, 50],
            'visible': 'all 400 - 700 nm',
            'all': 'all 51 bands'}

        self.n_classes = len(self.get_labels()) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        with open(osp.join(self.working_dir, set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()
        # self.ids = [splitext(file)[0] for file in listdir(self.specific_label_dir)
        #             if not file.startswith('.')]

    def __getitem__(self, index):
        # idx = self.ids[index]
        # rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.tif'))
        # rgb = rgb[:, :, ::-1]

        hsi = np.load(osp.join(self.working_dir, self.hsi_dir, self.filelist[index] + '.npy'))
        idx = self.filelist[index]
        # print('idx:',self.set_type,idx)
        if self.hsi_mode == 'visible':
            hsi = hsi[:, :, 0:31]
        elif self.hsi_mode == 'all':
            hsi = hsi
        else:
            bands = self.hsi_dict[self.hsi_mode]
            hsi_temp = np.zeros((hsi.shape[0], hsi.shape[1], len(bands)))
            for i in range(len(bands)):
                hsi_temp[:, :, i] = hsi[:, :, bands[i]]
            hsi = hsi_temp

        hsi = hsi.astype(np.float32)

        # label = imageio.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = label[:, :, ::-1]
        # print('-1 zhihou type', type(label))#<class 'numpy.ndarray'>
        # print('-1 zhihou', label.shape)#!!!!!!!!!!!!!!!!!!!!!!!1 (64, 64, 3)

        if self.augmentations is not None:
            hsi, label = self.augmentations(hsi, label)
            # print('augmentations type', type(label))#<class 'numpy.ndarray'>
            # print('augmentations', label.shape)# !!!!!!!!!!!!!!!!(64, 64, 3)

        if self.transforms is not None:
            # rgb = self.transforms(rgb)

            if self.hsi_sign == 'rad':
                # hsi = np.clip(hsi, 0, 2 ** 14) / 2 ** 14
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)  # np转为tensor
            elif self.hsi_sign == 'ref':
                # hsi = np.clip(hsi, 0, 100) / 100
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            label = self.encode_segmap(label)
            # print('encode_segmap type', type(label))#                                 <class 'numpy.ndarray'>
            # print('encode_segmap', label.shape)# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!(64, 64)
            label = torch.from_numpy(np.array(label)).long()
            # print('from_numpy type', type(label))#   <class 'torch.Tensor'>
            # print('from_numpy', label.shape)#        torch.Size([64, 64])
            # print('dtype',label.dtype)
            # print(label)
        # idex_category = self.get_idex_by_category(idx, label)

        # return hsi, label
        return {'hsi': hsi,
                'label': label,
                'idx': idx}

    def __len__(self):
        return len(self.filelist)

    # def get_idex_by_category(self, idx, label):
    #     label = label.numpy()
    #     if not(np.any(label)):
    #
    #             f.write("%s\n" % idx)
            # print()
        # return index

    def get_labels(self):
        return np.asarray(
            [
                [0,   0,   0],
                [0, 139,   0],
                [0,   0, 255],
                [255, 255, 0],
                [0, 255, 255],
                [255, 0, 255],
                [139, 139, 0],
                [0, 139, 139],
                [0, 255,   0],
                [0,   0, 139],#10
                [255, 127,80],
                [127, 255, 0],
                [218,112,214],
                [46, 139, 87],
                [131 ,111, 255],#15
                [255, 165, 0],#16hang

            ]
        )

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return np.uint8(rgb)

def parse_args(parser):
    '''
    Standard argument parser
    '''
    args = parser.parse_args()
    if args.config_file and os.path.exists(args.config_file):
        data = yaml.safe_load(open(args.config_file))
        delattr(args, 'config_file')
        arg_dict = args.__dict__
#        print (data)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args

AeroCLoader_camp = np.asarray(
                [
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                        [0, 255, 255],
                        [255, 127, 80],
                        [153, 0, 0],
                        ]
                )

XiongAnLoader_camp = np.asarray(
            [
                [0,   0,   0],
                [0, 139,   0],
                [0,   0, 255],
                [255, 255, 0],# 3
                [0, 255, 255],#4
                [255, 0, 255],
                [139, 139, 0],# 6
                [0, 139, 139],
                [0, 255,   0],#
                [0,   0, 139],#9
                [255, 127,80],
                [127, 255, 0],#11-5612 2.62%
                [218,112,214],
                [46, 139, 87],
                [131 ,111, 255],#14-7151
                [255, 165, 0],
                [127,255,212],
                # [196, 196, 0],#17-1496系柳林原本为：(218,112,214)和玉米颜色重了
                [255,  0,  0],
                [205,  0,  0],
                [139, 0, 0],
            ]
        )
LongKouLoader_camp = np.asarray(
            [
                [0, 0, 0],
                [0, 139, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],  # 5
                [255, 0, 255],
                [139, 139, 0],
                [0, 139, 139],
                [0, 255, 0],
                [0, 0, 139],  # 10
            ]
        )
HongHuLoader_camp = np.asarray(
            [
                [0,   0,   0],# 0
                [0, 139,   0],# 1
                [0,   0, 255],
                [255, 255, 0],
                [0, 255, 255],#4
                [255, 0, 255],
                [139, 139, 0],
                [0, 139, 139],
                [0, 255,   0],
                [0,   0, 139],#19
                [255, 127,80],
                [127, 255, 0],
                [218,112,214],
                [46, 139, 87],
                [131 ,111, 255],#14
                [255, 165, 0],
                [127,255,212],
                [196, 196, 0],
                [255,  0,  0],
                [205,  0,  0],#19
                [139,  0,  0],
                [255, 205, 0],
                [139, 205, 139]#22
            ]
        )
HanChuanLoader_camp = np.asarray(
            [
                [0, 0,   0], # 0
                [0, 139, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],  # 4
                [255, 0, 255],
                [139, 139, 0],
                [0, 139, 139],
                [0, 255, 0],
                [0, 0, 139],  # 9
                [255, 127, 80],
                [127, 255, 0],
                [218, 112, 214],
                [46, 139, 87],
                [131 ,111, 255],#14
                [255, 165, 0],
                [127, 255, 212],
            ]
        )