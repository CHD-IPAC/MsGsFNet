# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import argparse
import numpy as np
import torch
import torch.utils.data as data
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image

from torch import optim as optim
from helpers.utils_into_erf import AeroCLoader, XiongAnLoader, Metrics, parse_args, \
    ShandongDowntownLoader, LongKouLoader, HongHuLoader, HanChuanLoader
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--weights', default='//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestMiou_//HongHu//reconv19_normsmall in rate of 25.pt', type=str, help='path to weights file. For resnet101/152, ignore this arg to download from torchvision')
    parser.add_argument('--save_path', default='//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//reconv19_normsmall.png', type=str, help='path to save the ERF matrix (.npy file)')
    parser.add_argument('--num_images', default=50, type=int, help='num of images to use')   #  指定num_images，计算多少张图片的热力图的平均值，并保存
    parser.add_argument('--bands', default=270, help='Which bands category to load \
                            - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type=int)
    parser.add_argument('--num_class', default=23, help='The number of categories of the data', type=int)
    parser.add_argument('--hsi_c', default='HongHu', help='Load HSI Radiance or Reflectance data?')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of images sampled per minibatch?')
    parser.add_argument('--model', default='small', help='Network a  rchitecture?')
    args = parser.parse_args()
    return args

# 要注意，这个outputs可能是没有上采样的，求的是输出中心点对输入所有点的梯度，然后把这些梯度在0,1维度上加起来（也就是汇成一个（h，w））作为最终输出。
def get_input_grad(model, samples):
    outputs, _ = model(samples)
    out_size = outputs.size()     # out_size 是一个包含 outputs 尺寸的元组，例如 (batch_size, num_channels, height, width)。
    # 应用ReLU激活函数，将所有负值替换为0，将逐样本逐通道的最中间元素求和
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    # 计算 central_point 对输入 samples 的梯度
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]   # grad只有一个值，形状和samples一致
    # grad[i, j, k, l] 表示 central_point 对 samples[i, j, k, l] 的偏导数，反映了输入样本中该位置的变化对输出中间位置元素的影响程度。
    grad = torch.nn.functional.relu(grad)
    # 将梯度从 (batch_size, num_channels, height, width) 求和压缩到 (height, width)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map

def main(args):
    #   ================================= transform: resize to 1024x1024
    t = [
        # transforms.Resize((256, 256), interpolation=Image.BICUBIC),    # 不应该是1024吧？
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(t)

    print("reading from datapath")
    hsi_mode = 'all'
    if args.hsi_c == 'rad' or args.hsi_c == 'ref':
        trainset = AeroCLoader(set_loc='left', set_type='train', size='small',
                               hsi_sign=args.hsi_c, hsi_mode=hsi_mode, transforms=transform, augs=None)
        valset = AeroCLoader(set_loc='mid', set_type='test', size='small',
                             hsi_sign=args.hsi_c, hsi_mode=hsi_mode, transforms=transform)
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    elif args.hsi_c == 'HanChuan':
        trainset = HanChuanLoader(set_type='seed10_25_train_aug', hsi_mode='all', transforms=transform,
                                  augs=None)  ## first_train second_train third_train fourth_train fifth_train
        valset = HanChuanLoader(set_type='seed10_25_test', hsi_mode='all', transforms=transform)
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    elif args.hsi_c == 'HongHu':
        trainset = HongHuLoader(set_type='seed10_25_train_aug', hsi_mode='all', transforms=transform,
                                augs=None)  # first_train second_train third_train fourth_train fifth_train
        valset = HongHuLoader(set_type='seed10_25_test', hsi_mode='all', transforms=transform)
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    elif args.hsi_c == 'LongKou':
        trainset = LongKouLoader(set_type='seed10_25_train_aug', hsi_mode='all', transforms=transform,
                                 augs=None)  # first_train second_train third_train fourth_train fifth_train
        valset = LongKouLoader(set_type='seed10_25_test', hsi_mode='all', transforms=transform)
        ## first_val second_val third_val fourth_val fifth_val
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'SDA_dilation_stages_without_loss_ERF':
       from networks.SDA_dilation_stages_without_loss_ERF import ResnetGenerator
       model = ResnetGenerator(args.bands, args.num_class)

    elif args.model == 'small':
        from networks.MsGsFNet import MsGsFNet

        model = MsGsFNet(args.bands, args.num_class, False, 64)

    elif args.model == 'CLMGNet':
        from networks.CLMGNet import CLMGNet

        model = CLMGNet(args.num_class, args.bands, 4, 64)

    elif args.model == 'small_ablation_only_spatial':
        from networks.small_ablation_only_spatial import ResnetGenerator

        model = ResnetGenerator(args.bands, args.num_class, False, 256)

    elif args.model == 'CLMGNet_ablation_only_spatial':
        from networks.CLMGNet_ablation_only_spatial import CLMGNet

        model = CLMGNet(args.num_class, args.bands, 13, 256)

    elif args.model == 'lsk_ablation_only_spatial':
        from networks.lsk_ablation_only_spatial import ResnetGenerator

        model = ResnetGenerator(args.bands, args.num_class, False, 256)

    else:
       raise ValueError('Unsupported model. Please add it here.')

    if args.weights is not None:
        print('load weights')
        weights = torch.load(args.weights, map_location='cpu')
        if 'model' in weights:
            weights = weights['model']
        if 'state_dict' in weights:
            weights = weights['state_dict']
        model.load_state_dict(weights)
        print('loaded')

    model.cuda()
    model.eval()
    # if args.model == 'small' or model == 'small_addsmallkernel':
    #     model.structural_reparam()
    ##################################################################
    conv_weights = model.LSKBlock.attn.dwsmallkernel.lk_origin.weight.data.cpu().numpy()  # 提取权重
    # conv_weights = model.LSKBlock.attn.dwsmallkernel.weight.data.cpu().numpy()  # 提取权重
    # conv_weights 的形状为 (C_out, C_in, 19, 19)

    # 对通道进行绝对值求和聚合（aggregate across channels）
    aggregated_kernel = np.sum(np.abs(conv_weights), axis=(0, 1))  # 跨输出通道和输入通道求和
    # 结果形状为 (19, 19)

    # 归一化到 [0, 1]
    normalized_kernel = aggregated_kernel / aggregated_kernel.max()

    plt.figure(figsize=(8, 8))  # 图像大小
    ax = sns.heatmap(
        normalized_kernel,
        xticklabels=False,
        yticklabels=False,
        cmap='RdYlGn',  # 使用自定义颜色方案
        center=0,  # 对称居中
        annot=True,  # 显示数值
        cbar=True,  # 显示颜色条
        annot_kws={"size": 10},  # 数值字体大小
        fmt='.2f',# 数值格式
        square=True
    )  # 热力图
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, direction='out')  # 设置颜色条刻度字体大小
    cbar.ax.xaxis.set_ticks_position('top')  # 将刻度移到顶部
    cbar.ax.xaxis.set_label_position('top')  # 将标签移到顶部
    # cbar.ax.set_position([0.25, 0.9, 0.5, 0.02])  # 调整颜色条位置和大小：参数分别是 [left, bottom, width, height]
    plt.show()
    # plt.savefig(args.save_path)
    # print('already saved')
    ##################################################################

    # optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
    #
    # meter = AverageMeter()  # 用于记录平均梯度图
    # optimizer.zero_grad()
    #
    # for idx, batch in enumerate(valloader, 0):      # 第一个_是idx，第二个是labels
    #
    #     if meter.count == args.num_images:      # count 是计数器，表示处理的样本数量
    #         np.save(args.save_path, meter.avg)
    #         exit()
    #     samples = batch['hsi'].cuda(non_blocking=True)
    #     samples.requires_grad = True
    #     optimizer.zero_grad()
    #     contribution_scores = get_input_grad(model, samples)
    #
    #     if np.isnan(np.sum(contribution_scores)):
    #         print('got NAN, next image')
    #         continue
    #     else:
    #         print('accumulate')
    #         # 使用 meter.update(contribution_scores) 将每个样本的贡献分数累积到 AverageMeter 中。AverageMeter 会更新它的平均值和计数器
    #         meter.update(contribution_scores)



if __name__ == '__main__':
    args = parse_args()
    main(args)