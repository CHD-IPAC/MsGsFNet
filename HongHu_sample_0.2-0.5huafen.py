#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Wenqing Zhou
"""

import os
import os.path as osp
import random

import imageio
import numpy as np

from skimage import io
from PIL import Image
import cv2


def create_splits(percent, loc, size_chips, labels):

    # os.makedirs(loc, exist_ok = True)
    print('Starting chip making now')
    x_arr, y_arr = labels.shape #(1920,3968,3) # <class 'numpy.ndarray'>
    # print(type(labels))
    img2 = labels
    percent = percent
    percent2 = percent
    train_3 = open(osp.join(loc, 'NEW_seed10_{}_train.txt'.format(
        int(percent * 100))), 'w')
    test_3 = open(osp.join(loc, 'NEW_seed10_{}_test.txt'.format(
        int(percent * 100))), 'w')
    lei0zongshu = np.sum(img2 == 0)
    lei1zongshu = np.sum(img2 == 1)
    lei2zongshu = np.sum(img2 == 2)
    lei3zongshu = np.sum(img2 == 3)
    lei4zongshu = np.sum(img2 == 4)
    lei5zongshu = np.sum(img2 == 5)
    lei6zongshu = np.sum(img2 == 6)
    lei7zongshu = np.sum(img2 == 7)
    lei8zongshu = np.sum(img2 == 8)
    lei9zongshu = np.sum(img2 == 9)
    lei10zongshu = np.sum(img2 == 10)
    lei11zongshu = np.sum(img2 == 11)
    lei12zongshu = np.sum(img2 == 12)
    lei13zongshu = np.sum(img2 == 13)
    lei14zongshu = np.sum(img2 == 14)
    lei15zongshu = np.sum(img2 == 15)
    lei16zongshu = np.sum(img2 == 16)
    lei17zongshu = np.sum(img2 == 17)
    lei18zongshu = np.sum(img2 == 18)
    lei19zongshu = np.sum(img2 == 19)
    lei20zongshu = np.sum(img2 == 20)
    lei21zongshu = np.sum(img2 == 21)
    lei22zongshu = np.sum(img2 == 22)
    print('0出现的次数', np.sum(img2 == 0))
    print('1出现的次数', np.sum(img2 == 1))
    print('2出现的次数', np.sum(img2 == 2))
    print('3出现的次数', np.sum(img2 == 3))
    print('4出现的次数', np.sum(img2 == 4))
    print('5出现的次数', np.sum(img2 == 5))
    print('6出现的次数', np.sum(img2 == 6))
    print('7出现的次数', np.sum(img2 == 7))
    print('8出现的次数', np.sum(img2 == 8))
    print('9出现的次数', np.sum(img2 == 9))
    print('10出现的次数', np.sum(img2 == 10))
    print('11出现的次数', np.sum(img2 == 11))
    print('12出现的次数', np.sum(img2 == 12))
    print('13出现的次数', np.sum(img2 == 13))
    print('14出现的次数', np.sum(img2 == 14))
    print('15出现的次数', np.sum(img2 == 15))
    print('16出现的次数', np.sum(img2 == 16))
    print('17出现的次数', np.sum(img2 == 17))
    print('18出现的次数', np.sum(img2 == 18))
    print('19出现的次数', np.sum(img2 == 19))
    print('20出现的次数', np.sum(img2 == 20))
    print('21出现的次数', np.sum(img2 == 21))
    print('22出现的次数', np.sum(img2 == 22))
    for lei in range(23):
        globals()[f'trainlei{lei}'] = 0
    testc0 = 0
    testc1 = 0
    testc2 = 0
    testc3 = 0
    testc4 = 0
    testc5 = 0
    testc6 = 0
    testc7 = 0
    testc8 = 0
    testc9 = 0
    testc10 = 0
    testc11 = 0
    testc12 = 0
    testc13 = 0
    testc14 = 0
    testc15 = 0
    testc16 = 0
    testc17 = 0
    testc18 = 0
    testc19 = 0
    testc20 = 0
    testc21 = 0
    testc22 = 0

    pic = 0
    pic_test = 0
    pic0 = 0
    picnot0 = 0
    chunbeijing = 0
    name_saved = []
    for xx in range(0, x_arr - size_chips//2, size_chips):
        for yy in range(0, y_arr - size_chips//2, size_chips):
            name = 'image_{}_{}'.format(xx, yy)
            name_saved.append(name)
    random.seed(10)
    random.shuffle(name_saved)
    print('name_saved长度：', len(name_saved))
    for i in range(1, len(name_saved)+1):
        name = name_saved[i-1]
        label_one_chann = imageio.v2.imread(osp.join(loc, 'one_chann_Labels', name + '.tif'))  # yongyu panduan
        unique_values = np.unique(label_one_chann)
        if np.any(label_one_chann):  # if not all ground  至少有一类了 排除纯背景
            pic = pic + 1
            # print('_pic:',pic) #405
            # 15,21,17, 2,18,20,22,8, 5,14,16,12,19,9, 11,1,10,3,13, 7, 6, 4
            if (15 in unique_values) and globals()['trainlei15'] <= int(lei15zongshu * percent2):
                # print('百分比',lei15zongshu * percent2)
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                # print('trainlei15:--',trainlei15)
                train_3.write("%s\n" % name)
                continue
            elif (21 in unique_values) and globals()['trainlei21'] <= int(lei21zongshu * percent2):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (17 in unique_values) and globals()['trainlei17'] <= int(lei17zongshu * percent2):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (2 in unique_values) and globals()['trainlei2'] <= int(lei2zongshu * percent2):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (18 in unique_values) and globals()['trainlei18'] <= int(lei18zongshu * percent2):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (20 in unique_values) and globals()['trainlei20'] <= int(lei20zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (22 in unique_values) and globals()['trainlei22'] <= int(lei22zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (8 in unique_values) and globals()['trainlei8'] <= int(lei8zongshu * percent2):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (5 in unique_values) and globals()['trainlei5'] <= int(lei5zongshu * percent2):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (14 in unique_values) and globals()['trainlei14'] <= int(lei14zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (16 in unique_values) and globals()['trainlei16'] <= int(lei16zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (12 in unique_values) and globals()['trainlei12'] <= int(lei12zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (19 in unique_values) and globals()['trainlei19'] <= int(lei19zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (9 in unique_values) and globals()['trainlei9'] <= int(lei9zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (11 in unique_values) and globals()['trainlei11'] <= int(lei11zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (1 in unique_values) and globals()['trainlei1'] <= int(lei1zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (10 in unique_values) and globals()['trainlei10'] <= int(lei10zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (3 in unique_values) and globals()['trainlei3'] <= int(lei3zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (13 in unique_values) and globals()['trainlei13'] <= int(lei13zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (7 in unique_values) and globals()['trainlei7'] <= int(lei7zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (6 in unique_values) and globals()['trainlei6'] <= int(lei6zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (4 in unique_values) and globals()['trainlei4'] <= int(lei4zongshu * percent):
                for lei in range(23):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            else:
                pic_test = pic_test + 1
                # print('pic_test:',pic_test)
                test_3.write("%s\n" % name)
                # testpath = osp.join(loc, 'one_chann_Labels', name + '.tif')  # yongyu panduan
                # img2 = Image.open(testpath)
                # img2 = np.array(img2)
                img2 = label_one_chann
                testc0 = testc0 + np.sum(img2 == 0)
                testc1 = testc1 + np.sum(img2 == 1)
                testc2 = testc2 + np.sum(img2 == 2)
                testc3 = testc3 + np.sum(img2 == 3)
                testc4 = testc4 + np.sum(img2 == 4)
                testc5 = testc5 + np.sum(img2 == 5)
                testc6 = testc6 + np.sum(img2 == 6)
                testc7 = testc7 + np.sum(img2 == 7)
                testc8 = testc8 + np.sum(img2 == 8)
                testc9 = testc9 + np.sum(img2 == 9)
                testc10 = testc10 + np.sum(img2 == 10)
                testc11 = testc11 + np.sum(img2 == 11)
                testc12 = testc12 + np.sum(img2 == 12)
                testc13 = testc13 + np.sum(img2 == 13)
                testc14 = testc14 + np.sum(img2 == 14)
                testc15 = testc15 + np.sum(img2 == 15)
                testc16 = testc16 + np.sum(img2 == 16)
                testc17 = testc17 + np.sum(img2 == 17)
                testc18 = testc18 + np.sum(img2 == 18)
                testc19 = testc19 + np.sum(img2 == 19)
                testc20 = testc20 + np.sum(img2 == 20)
                testc21 = testc21 + np.sum(img2 == 21)
                testc22 = testc22 + np.sum(img2 == 22)

        else:
            chunbeijing = chunbeijing + 1
            # print('chunbeijing', chunbeijing)
    # print('testchunc0', testchunc0)
    print('pic_test', pic_test)
    print('trainlei0', trainlei0)
    print('trainlei1', trainlei1)
    print('trainlei2', trainlei2)
    print('trainlei3', trainlei3)
    print('trainlei4', trainlei4)
    print('trainlei5', trainlei5)
    print('trainlei6', trainlei6)
    print('trainlei7', trainlei7)
    print('trainlei8', trainlei8)
    print('trainlei9', trainlei9)
    print('trainlei10', trainlei10)
    print('trainlei11', trainlei11)
    print('trainlei12', trainlei12)
    print('trainlei13', trainlei13)
    print('trainlei14', trainlei14)
    print('trainlei15', trainlei15)
    print('trainlei16', trainlei16)
    print('trainlei17', trainlei17)
    print('trainlei18', trainlei18)
    print('trainlei19', trainlei19)
    print('trainlei20', trainlei20)
    print('trainlei21', trainlei21)
    print('trainlei22', trainlei22)
    print('testc0', testc0)
    print('testc1', testc1)
    print('testc2', testc2)
    print('testc3', testc3)
    print('testc4', testc4)
    print('testc5', testc5)
    print('testc6', testc6)
    print('testc7', testc7)
    print('testc8', testc8)
    print('testc9', testc9)
    print('testc10', testc10)
    print('testc11', testc11)
    print('testc12', testc12)
    print('testc13', testc13)
    print('testc14', testc14)
    print('testc15', testc15)
    print('testc16', testc16)
    print('testc17', testc17)
    print('testc18', testc18)
    print('testc19', testc19)
    print('testc20', testc20)
    print('testc21', testc21)
    print('testc22', testc22)
    # print(chunbeijing)    1

    for i in range(23):
        trainlei = f'trainlei{i}'
        print(trainlei, globals()[trainlei]/np.sum(labels == i))

    print('相加的结果:', testc1 + trainlei1 == lei1zongshu)

    print('Stopping chip making now')

if __name__ == "__main__":
    
    folder_dir = osp.join('Collection')  # path to full files

    labels = io.imread(osp.join(folder_dir, 'WHU-Hi-HongHu_gt.tif'))[12:,27:,]
    # (928, 448)
    image1_labels = labels[:, :]

    # total_pixels = labels.size
    # background_pixels = np.sum(labels == 0)
    # valid_pixels = total_pixels - background_pixels
    #
    # valid_ratio = valid_pixels / total_pixels * 100  # 计算有效标注比例（百分比）
    # print(f"有效标注比例：{valid_ratio:.2f}%")  # 88.11%

    percent = 0.25

    patch_size = 32

    create_splits(percent,
                  loc=osp.join('Image{}_step_patch'.format(patch_size)),
                  size_chips=patch_size,
                  labels=image1_labels,
                  )


    def process_txt_file(file_path, output_file_path):
        # 确保输入路径和输出路径格式正确
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # 读取文件的所有行

        if not lines:  # 如果文件为空或读取失败，提示错误
            print("Error: Failed to read lines from the input file.")
            return

        new_lines = []  # 存储修改后的内容

        # 1. 先添加原始内容
        for line in lines:
            line = line.strip()  # 去掉行末的换行符和空格
            new_lines.append(f"{line}")  # 原始内容不做修改

        # 2. 添加 "_Horizon" 后缀的内容
        for line in lines:
            line = line.strip()
            new_lines.append(f"{line}_Horizon")

        # 3. 添加 "_Vertically" 后缀的内容
        for line in lines:
            line = line.strip()
            new_lines.append(f"{line}_Vertically")

        # 4. 添加 "_Transpose" 后缀的内容
        for line in lines:
            line = line.strip()
            new_lines.append(f"{line}_Transpose")

        # 将新的内容写入新的txt文件
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for new_line in new_lines:
                output_file.write(f"{new_line}\n")

        print(f"File processed successfully and saved to {output_file_path}")


    # 调用函数
    process_txt_file(
        r'E:\Something\oold-take new loss-XiongAn-master_noleft_concision_two 7×7s\WHU-Hi-HongHu\Image32_step_patch\NEW_seed10_{}_train.txt'.format(int(percent*100)),
        r'E:\Something\oold-take new loss-XiongAn-master_noleft_concision_two 7×7s\WHU-Hi-HongHu\Image32_step_patch\NEW_seed10_{}_train_aug.txt'.format(int(percent*100)))
