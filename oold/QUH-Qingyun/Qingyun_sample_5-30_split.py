from scipy.io import loadmat
"""
@author: Wenqing Zhou
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
    x_arr, y_arr = labels.shape
    # print(np.unique(labels))   [4, 3, 2, 5, 0, 6, 1]
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
    print('0出现的次数', np.sum(img2 == 0))
    print('1出现的次数', np.sum(img2 == 1))
    print('2出现的次数', np.sum(img2 == 2))
    print('3出现的次数', np.sum(img2 == 3))
    print('4出现的次数', np.sum(img2 == 4))
    print('5出现的次数', np.sum(img2 == 5))
    print('6出现的次数', np.sum(img2 == 6))

######################################################################################
    occurrences = []
    # 计算每个类别出现的次数并打印
    for i in range(7):
        count = np.sum(img2 == i)
        occurrences.append((i, count))  # 将类别号和出现次数存储起来
    # 按出现次数从小到大排序
    sorted_occurrences = sorted(occurrences, key=lambda x: x[1])
    # 提取排序后的类号
    sorted_classes = [x[0] for x in sorted_occurrences]
    # 打印排序后的类号
    print('从小到大排序后的类号:', sorted_classes)
######################################################################################
    for lei in range(7):
        globals()[f'trainlei{lei}'] = 0
    testc0 = 0
    testc1 = 0
    testc2 = 0
    testc3 = 0
    testc4 = 0
    testc5 = 0
    testc6 = 0

    pic = 0
    pic_test = 0
    pic0 = 0
    picnot0 = 0
    chunbeijing = 0
    name_saved = []
    for xx in range(0, x_arr - size_chips // 2, size_chips):
        for yy in range(0, y_arr - size_chips // 2, size_chips):
            name = 'image_{}_{}'.format(xx, yy)
            name_saved.append(name)
    random.seed(10)  # 1  4  5  6
    random.shuffle(name_saved)
    print('name_saved长度：', len(name_saved))
    for i in range(1, len(name_saved) + 1):
        name = name_saved[i - 1]
        label_one_chann = imageio.v2.imread(osp.join(loc, 'one_chann_Labels', name + '.tif'))  # yongyu panduan
        unique_values = np.unique(label_one_chann)
        if np.any(label_one_chann):  # if not all ground  至少有一类了 排除纯背景
            pic = pic + 1
            # print('_pic:',pic) #405
            # 15,21,17, 2,18,20,22,8, 5,14,16,12,19,9, 11,1,10,3,13, 7, 6, 4
            if (4 in unique_values) and globals()['trainlei4'] <= int(lei4zongshu * percent2):
                # print('百分比',lei15zongshu * percent2)
                for lei in range(11):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                # print('trainlei15:--',trainlei15)
                train_3.write("%s\n" % name)
                continue
            elif (3 in unique_values) and globals()['trainlei3'] <= int(lei3zongshu * percent2):
                for lei in range(11):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (2 in unique_values) and globals()['trainlei2'] <= int(lei2zongshu * percent2):
                for lei in range(11):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (5 in unique_values) and globals()['trainlei5'] <= int(lei5zongshu * percent2):
                for lei in range(11):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (6 in unique_values) and globals()['trainlei6'] <= int(lei6zongshu * percent2):
                for lei in range(11):  # 假设有 23 个类别
                    if lei in unique_values:
                        globals()[f'trainlei{lei}'] += np.sum(label_one_chann == lei)
                train_3.write("%s\n" % name)
                continue
            elif (1 in unique_values) and globals()['trainlei1'] <= int(lei1zongshu * percent):
                for lei in range(11):  # 假设有 23 个类别
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
    print('testc0', testc0)
    print('testc1', testc1)
    print('testc2', testc2)
    print('testc3', testc3)
    print('testc4', testc4)
    print('testc5', testc5)
    print('testc6', testc6)
    print(chunbeijing)

    for i in range(7):
        trainlei = f'trainlei{i}'
        print(trainlei, globals()[trainlei] / np.sum(labels == i))

    print('相加的结果:', testc1 + trainlei1 == lei1zongshu)

    print('Stopping chip making now')


if __name__ == "__main__":
    labels1 = loadmat(
        'E:\Something\oold-take new loss-XiongAn-master_noleft_concision_two 7×7s\QUH-Qingyun\Collection\QUH-Qingyun_GT.mat')
    labels = labels1['ChengquGT'][:864, :1344]
    # folder_dir = osp.join('Collection')  # path to full files
    #
    # labels = io.imread(osp.join(folder_dir, 'WHU-Hi-HongHu_gt.tif'))[12:, 27:, ]
    # image1_labels = labels[:, :]

    # total_pixels = labels.size
    # background_pixels = np.sum(labels == 0)
    # valid_pixels = total_pixels - background_pixels
    #
    # valid_ratio = valid_pixels / total_pixels * 100  # 计算有效标注比例（百分比）
    # print(f"有效标注比例：{valid_ratio:.2f}%")    # 79.83%

    image1_labels = labels[:, :]

    percent = 0.25     # 0.25 means the 25%

    create_splits(percent,
                  loc=osp.join('Image32_step_patch'),
                  size_chips=32,
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
        r'E:\Something\oold-take new loss-XiongAn-master_noleft_concision_two 7×7s\QUH-Qingyun\Image32_step_patch\NEW_seed10_{}_train.txt'.format(
            int(percent * 100)),
        r'E:\Something\oold-take new loss-XiongAn-master_noleft_concision_two 7×7s\QUH-Qingyun\Image32_step_patch\NEW_seed10_{}_train_aug.txt'.format(
            int(percent * 100)))
