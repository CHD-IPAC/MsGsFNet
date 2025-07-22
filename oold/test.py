#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 此test文件是对已划分的test集中的所有小图片进行预测，所以输出的精度是对所有test集进行计算。
"""
Created on Wed Jul 10 22:56:02 2019

"""

import torch
from torchvision import transforms
import numpy as np
import os
from helpers.Metrics_two import metric, ConfusionMatrix
from helpers.utils import Metrics, AeroCLoader, parse_args, XiongAnLoader, ShandongDowntownLoader, HanChuanLoader, \
    HongHuLoader, LongKouLoader, TangdaowanLoader, QingyunLoader, PinganLoader
from timm.models.vision_transformer import _cfg
import argparse
import time
from helpers.utils_batch_based import patch_based_HongHuLoader
from thop import profile


def test_model(weights_path, model, dataset, bands, num_class, rate_trainset, save_pred, patch_based, rate_patchbased, inner_nc, net_reparam, patchsize):

    parser = argparse.ArgumentParser(description='AeroRIT baseline evalutions')

    ### 0. Config file?
    parser.add_argument('--config-file', default=None, help='Path to configuration file')

    ### 1. Data Loading
    parser.add_argument('--bands', default=bands, help='Which bands category to load \
                            - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type=int)
    parser.add_argument('--num_class', default=num_class, help='The number of categories of the data', type=int)
    parser.add_argument('--hsi_c', default=dataset, help='Load HSI Radiance or Reflectance data?')

    ### 2. Network selections
    ### a. Which network?
    parser.add_argument('--network_arch', default=model, help='Network architecture?')
    # parser.add_argument('--network_arch', default='MsGsFNet', help='Network architecture?')
    parser.add_argument('--use_mini', action='store_true', help='Use mini version of network?')

    ### b. ResNet config
    parser.add_argument('--resnet_blocks', default=6, help='How many blocks if ResNet architecture?', type=int)

    ### c. UNet configs
    parser.add_argument('--use_SE', default=True, action='store_true', help='Network uses SE Layer?')
    parser.add_argument('--use_preluSE', default=True, action='store_true',
                        help='SE layer uses ReLU or PReLU activation?')

    ## Load weights post network config
    parser.add_argument('--network_weights_path',
                        default=weights_path,
                        help='Path to Saved Network weights')
    # parser.add_argument('--network_weights_path',
    #                     default='E:\Something\pretrainedweightssavedmodels\\_bestoa_\\rad\\ final use UniRepLKNetBlock_avgmax_196.pt',
    #                     help='Path to Saved Network weights')
    # parser.add_argument('--network_weights_path',
    #                    default='E:\Something\pretrainedweightssavedmodels\\rad\\ final use UniRepLKNetBlock_avgmax_196.pt',
    #                   help='Path to Saved Network weights')
    # parser.add_argument('--network_weights_path',
    #                     default='E:\Something\pretrainedweightssavedmodels\\best results\\rad\\final use dilated_branches_avgmax.pt',
    #                     help='Path to Saved Network weights')
    ### Use GPU or not
    parser.add_argument('--use_cuda', action='store_true', default=True, help='use GPUs?')

    args = parse_args(parser)
    print(args)

    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if args.hsi_c == 'rad':
        perf = Metrics(ignore_index=5)
    else:
        perf = Metrics(ignore_index=0)

    tx = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 数据标准化
    ])
    if save_pred:
        test_type = 'seed_10_{}_whole_test'.format(rate_trainset)
    else:
        test_type = 'NEW_seed10_{}_test'.format(rate_trainset)
    # NEW_seed10_{}_test
    # seed_10_{}_whole_test
    if args.hsi_c == 'rad' or args.hsi_c == 'ref':
        # testset = AeroCLoader(set_loc='right', set_type='test', size='small', hsi_sign=args.hsi_c, hsi_mode='all',
        #                       transforms=tx)
        testset = AeroCLoader(set_type=test_type,
                             size='small',
                             hsi_sign=args.hsi_c, hsi_mode='all', transforms=tx)
    # 注意：seed_10_25_whole_test只在出最终预测图的时候使用，这是为了给所有图像打标签。
    # elif args.hsi_c == 'XiongAn':
    #     testset = XiongAnLoader(set_type=test_type, hsi_mode='all', transforms=tx)
    elif args.hsi_c == 'LongKou':
        testset = LongKouLoader(set_type=test_type, size=patchsize, hsi_mode='all', transforms=tx)
    elif args.hsi_c == 'HongHu':
        if patch_based:
            testset = patch_based_HongHuLoader(set_type='test_{}'.format(rate_patchbased), hsi_mode='all', transforms=tx)
        else:
            testset = HongHuLoader(set_type=test_type, size=patchsize, hsi_mode='all', transforms=tx)
    elif args.hsi_c == 'HanChuan':
        testset = HanChuanLoader(set_type=test_type, size=patchsize, hsi_mode='all', transforms=tx)
    # elif args.hsi_c == 'ShandongDowntown':
    #     testset = ShandongDowntownLoader(set_type='lei1', hsi_mode='all', transforms=tx)
    # elif args.hsi_c == 'Tangdaowan':
    #     testset = TangdaowanLoader(set_type=test_type, hsi_mode='all', transforms=tx)
    elif args.hsi_c == 'Qingyun':
        testset = QingyunLoader(set_type=test_type, size=patchsize, hsi_mode='all', transforms=tx)
    # elif args.hsi_c == 'Pingan':
    #     testset = PinganLoader(set_type=test_type, hsi_mode='all', transforms=tx)

    print('Completed loading data...')
    if patch_based:
        num_class = args.num_class-1
    else:
        num_class = args.num_class
    if args.network_arch == 'resnet':
        from networks.resnet6 import ResnetGenerator

        net = ResnetGenerator(args.bands, num_class, n_blocks=args.resnet_blocks)
    elif args.network_arch == 'Resnet6Ghostv2':
        from networks.resnet6_ghostv2 import Resnet6Ghostv2

        # from networks.resnet6_ghostv2_2 import Resnet6Ghostv2
        net = Resnet6Ghostv2(args.bands, num_class, n_blocks=args.resnet_blocks)
    elif args.network_arch == 'Resnet6efficientvit':
        from networks.resnet6_3efficientvit import ResnetGenerator

        # from networks.resnet6_4shortcut import ResnetGenerator
        net = ResnetGenerator(args.bands, num_class, n_blocks=args.resnet_blocks)
    elif args.network_arch == 'resnet6_4shortcut':
        # from networks.resnet6_4shortcut import ResnetGenerator
        # from networks.resnet6_5ASSP import ResnetGenerator
        # from networks.resnet6_5ASSP_shortcutse_11_3D_conv import ResnetGenerator
        from networks.resnet6_5ASSP_shortcutse_11 import ResnetGenerator

        # from networks.resnet6_5ASSP_shortcutse_11_downsamplingthree import ResnetGenerator
        net = ResnetGenerator(args.bands, num_class)
    elif args.network_arch == 'resnet6_4shortcut_gff':
        from networks.resnet6_5ASSP_shortcutse_11_gff import ResnetGenerator

        net = ResnetGenerator(args.bands, num_class)
    elif args.network_arch == 'resnet6_4shortcut_gff_UpsamplingB':
        from networks.resnet6_5ASSP_shortcutse_11_gff_UpsamplingB import ResnetGenerator

        net = ResnetGenerator(args.bands, num_class)
    elif args.network_arch == 'resnet6_4shortcut_gff_F_interpolate':
        from networks.resnet6_5ASSP_shortcutse_11_gff_F_interpolate import ResnetGenerator

        net = ResnetGenerator(args.bands, num_class)

    elif args.network_arch == 'lskandreparamed':
        from networks.lskandreparamed import ResnetGenerator

        net = ResnetGenerator(args.bands,
                              num_class)
    elif args.network_arch == 'SegNext':
        from SegNext_master.model import SegNext
        net = SegNext(in_channnels=args.bands, num_classes=num_class)

    elif args.network_arch == 'GSCVIT':
        if patch_based:
            from networks.original_gscvit import gscvit
        else:
            from networks.gscvit import gscvit
        net = gscvit(dataset=args.hsi_c)
        net.default_cfg = _cfg()

    elif args.network_arch == 'DBCTNet':
        if patch_based:
            from networks.orginal_DBCTNet import DBCTNet
        else:
            from networks.DBCTNet import DBCTNet
        if args.hsi_c == 'rad':
            patch = 64
        else:
            patch = 32
        net = DBCTNet(bands=args.bands, num_class=num_class, patch=patch)

    elif args.network_arch == 'CLMGNet':
        if patch_based:
            from networks.original_CLMGNet import CLMGNet
        else:
            from networks.CLMGNet import CLMGNet
        net = CLMGNet(num_class, args.bands, 13, 256)

    elif args.network_arch == 'CLMGNet_ablation_only_spatial':
        from networks.CLMGNet_ablation_only_spatial import CLMGNet

        net = CLMGNet(num_class, args.bands, 13, 256)

    elif args.network_arch == 'MsGsFNet_baseline':
        from networks.MsGsFNet_baseline import MsGsFNet

        net = MsGsFNet(args.bands, num_class, patch_based, inner_nc)

    elif args.network_arch == 'FullyContNet':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        parser1 = argparse.ArgumentParser(description="test")
        args1 = parser1.parse_args()
        args1.network = 'FContNet'
        args1.head = 'psp'
        args1.mode = 'p_s_c'
        args1.input_size = [32, 32]
        args1.network = 'FContNet'
        from networks.FullyContNet import fucontnet

        net = fucontnet(args1, args.bands, num_class).cuda()

    elif args.network_arch == 'lsk_ablation_only_spatial':
        from networks.lsk_ablation_only_spatial import ResnetGenerator

        net = ResnetGenerator(args.bands, num_class, patch_based, 256)

    elif args.network_arch == 'resnet6_5ASSP_shortcutse_11_gff_F_interpolate_Ablation':
        from networks.resnet6_5ASSP_shortcutse_11_gff_F_interpolate_Ablation import ResnetGenerator

        net = ResnetGenerator(args.bands, num_class)
    elif args.network_arch == 'MACUNet':
        from networks.MACUNet import MACUNet

        net = MACUNet(args.bands, num_class)
    elif args.network_arch == 'SS3FCN':
        from networks.SS3FCN import SS3FCN
        if args.hsi_c == 'Qingyun' or args.hsi_c == 'Tangdaowan':
            net = SS3FCN(num_class, QUH=True)
        else:
            net = SS3FCN(num_class)
    elif args.network_arch == 'SS3FCN_rad':
        from networks.SS3FCN_rad import SS3FCN

        net = SS3FCN(args.bands, num_class)
    elif args.network_arch == 'segnetm':
        from networks.segnet import segnet, segnetm

        net = segnetm(args.bands, num_class)
        # if args.use_mini == True:
        #     net = segnetm(args.bands, num_class)
        # else:
        #     net = segnet(args.bands, num_class)
    elif args.network_arch == 'unetm':
        from networks.unet import unet, unetm

        # from networks.unet_ASPP import unet, unetm
        net = unetm(args.bands, num_class, use_SE=args.use_SE, use_PReLU=args.use_preluSE)
        # if args.use_mini == True:
        #     net = unetm(args.bands, num_class, use_SE = args.use_SE, use_PReLU = args.use_preluSE)
        # else:
        #     net = unet(args.bands, num_class)
    elif args.network_arch == 'FCN8':
        # from networks.FCN8 import FCN8
        from networks.FCN8_tiny import FCN8

        net = FCN8(args.bands, num_class)
    elif args.network_arch == 'DeepLabV3':
        from networks.Deeplab.deeplabv3 import DeepLabV3

        net = DeepLabV3(args.bands, num_class)

    elif args.network_arch == 'MsGsFNet':
        from networks.MsGsFNet import MsGsFNet
        net = MsGsFNet(args.bands, num_class, patch_based, inner_nc)

    elif args.network_arch == 'small_ablation_only_spatial':
        from networks.small_ablation_only_spatial import ResnetGenerator

        net = ResnetGenerator(args.bands, num_class, patch_based, 64)

    elif args.network_arch == 'small_addsmallkernel':
        from networks.small_addsmallkernel import ResnetGenerator

        net = ResnetGenerator(args.bands, num_class, patch_based, inner_nc)

    elif args.network_arch == 'small_lsk':
        from networks.small_lsk import ResnetGenerator

        net = ResnetGenerator(args.bands,
                              num_class)

    elif args.network_arch == 'DeepLabV3_plus':
        if args.hsi_c == 'rad':
            from networks.Deeplab.deeplabv3plus import DeepLabV3Plus
        else:
            from networks.Deeplab.deeplabv3plus_tiny import DeepLabV3Plus
        net = DeepLabV3Plus(args.bands, num_class)
        # from networks.deeplabv3_plus import DeepLab
        # net = DeepLab(args.bands, num_class)
    elif args.network_arch == 'LinkNet':
        from networks.LinkNet import LinkNet

        net = LinkNet(args.bands, num_class)
    elif args.network_arch == 'D-LinkNet':
        from networks.dinknet import DinkNet18

        net = DinkNet18(args.bands, num_class)
    elif args.network_arch == 'CNN_1D':
        from networks.CNN_1D import CNN_1D

        net = CNN_1D(args.bands, num_class)
    elif args.network_arch == 'CNN_3D':
        from networks.CNN_3D import CNN_3D

        net = CNN_3D(args.bands, num_class)

    elif args.network_arch == 'SegFormer':
        from networks.segformer import SegFormer
        net = SegFormer(dim=args.bands, num_classes=num_class, phi='b0')

    elif args.network_arch == 'repvit':

        from networks.repvit import repvit_m0_9

        net = repvit_m0_9(dim=args.bands, num_classes=num_class)

    elif args.network_arch == 'HPDM_SPRN':
        if patch_based:
            from networks.original_SPRN import HPDM_SPRN
        else:
            from networks.SPRN import HPDM_SPRN
        net = HPDM_SPRN(args.bands, num_class)

    elif args.network_arch == 'SSTN':
        if patch_based:
            from networks.original_sstn import SSNet_AEAE_UP
        else:
            from networks.SSTN import SSNet_AEAE_UP
        if args.network_arch == 'rad':
            net = SSNet_AEAE_UP(in_dim=args.bands, msize=32, num_classes=num_class, inter_size=64)
        else:
            net = SSNet_AEAE_UP(in_dim=args.bands, msize=64, num_classes=num_class, inter_size=128)

    elif args.network_arch == 'S3ANet':
        from networks.Model_S3ANet import S3ANet

        net = S3ANet(args.bands, num_class)

    elif args.network_arch == 'FreeNet':
        from networks.FreeNet import FreeNet

        config = dict(
            in_channels=args.bands,
            num_classes=num_class,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0, )
        net = FreeNet(config)
        # net = FreeNet(args.bands, num_class)

    else:
        raise NotImplementedError('required parameter not found in dictionary')

    net.load_state_dict(torch.load(args.network_weights_path), strict=False)
    net.eval()
    net.to(device)
    if model == 'MsGsFNet':
        if net_reparam:
            net.structural_reparam()

    # import numpy as np
    # import matplotlib.pyplot as plt
    # import rasterio
    #
    # image_path = '/media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962/zwq/oold/WHU-Hi-HongHu/Collection/WHU-Hi-HongHu.tif'  # 替换为您的图像文件路径
    #
    # # 使用 rasterio 打开影像和标签
    # with rasterio.open(image_path) as image_dataset:
    #     image = image_dataset.read().astype(np.float32)  # 图像形状 (Bands, Height, Width)
    # # image = np.transpose(image, (2, 0, 1))
    # image = torch.from_numpy(image)
    # image_input = image.unsqueeze(0)
    # whole_pred, _ = net(image_input.cuda())

    # net.merge_dilated_branches()

    print('Completed loading pretrained network weights...')

    print('Calculating prediction accuracy...')
    test_leak_which_index = []
    labels_gt = []
    labels_pred = []
    matrix_ = [[0] * (num_class)] * (num_class)
    add = 0
    whole_time = []
    for idx, batch in enumerate(testset, 0):
        # print(idx) 0-405(honghu)
        add = add + 1
        hsi = batch['hsi'].unsqueeze(0).to(device)
        # print(hsi.shape) torch.Size([270, 32, 32])
        label = batch['label']
        # hsi, label = testset[img_idx]
        label = label.numpy()
        start = time.time()  # 计时开始
        label_pred, att = net(hsi)  # 这里的batch-size是1(新加的维度)
        end = time.time()  # 计时结束
        interval = end - start
        whole_time.append(interval)
        if idx == 0:
            flops, params = profile(net, inputs=(hsi,))

        # label_pred = label_pred.squeeze_(0).cpu().detach().numpy()
        # label_pred = np.transpose(label_pred, (1, 2, 0))
        label_pred = label_pred.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()  # .max(1)返回最大值和索引，[1]取索引，.squeeze_(1)是压缩通道维度，.squeeze_(0)是压缩batch维度
        # print('1',label_pred.shape)  (32, 32)
        if save_pred and patch_based==False:
            os.makedirs("//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//the pred pictures//{}//{}_{}".format(args.hsi_c, args.network_arch, args.hsi_c), exist_ok=True)
            filename = "//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//the pred pictures//{}//{}_{}//labels_pred_%d.npy".format(args.hsi_c, args.network_arch, args.hsi_c) % idx
        # 保存预测结果
            np.save(filename, label_pred)
        label = label.flatten()
        label_pred = label_pred.flatten()
        matrix = ConfusionMatrix(numClass=num_class, imgPredict=label_pred, Label=label, set=args.hsi_c)
        matrix_ += matrix
        labels_gt = np.append(labels_gt, label)  # 由此可以看出，这是把一个个小块进行预测，每一个像素逐通道打标签，然后拿出这些个通道标签的最大值作为最后的结果。然后一个个小图的标签都展平（第一行后紧跟第二行），放入列表。
        # print('labels_gt',labels_gt)
        # unique_values = np.unique(labels_gt)
        # print('unique_values',unique_values)
        # test_leak_which_index.extend(unique_values)
        # mylist = list(set(test_leak_which_index))
        # print('test_leak_which_index:', mylist)
        labels_pred = np.append(labels_pred, label_pred)
        max_idx = -1
        max_idx = max(max_idx, idx) + 1
    npy_save_path = './savedmodels/' + args.hsi_c + '/' + args.network_arch + '/' + 'confusion_matrix_plot/'
    os.makedirs(npy_save_path, exist_ok=True)
    np.save(npy_save_path + "labels_gt.npy", labels_gt)
    np.save(npy_save_path + "labels_pred.npy", labels_pred)
    # np.save("./savedmodels/HongHu/resnet6_4shortcut/fourth_labels_gt.npy", labels_gt)
    # np.save("./savedmodels/HongHu/resnet6_4shortcut/fouth_labels_pred.npy", labels_pred)

    # print('zuida index', np.max(labels_gt))  # 19 ??
    print('unique index', np.unique(labels_gt))  # 19 ??
    # print('first hunxiaojuzhen +delai shape:', matrix_.shape)  # first hunxiaojuzhen shape: (17, 17)
    # print('first hunxiaojuzhen +delai:\n',matrix_)
    # precision, recall, OA, IoU, FWIOU, mIOU, f1score = metric(matrix_)
    # print('OA: {}\nFWIoU: {}\nmIoU: {}\n'.format(OA, FWIOU, mIOU))
    # print('\nOA: {}\nmIoU: {}\nFWIoU: {}'.format(OA, mIOU, FWIOU))
    # print('IoU: {}'.format(IoU))
    # print('f1score: {}'.format(f1score))
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    print("\n**********************************OA*AA*Miou*DICE**************")
    print('Data set:                                                ', args.hsi_c)
    print('The path of saved weights:                               ', args.network_weights_path)

    scores = perf(labels_gt, labels_pred)
    save_dir = os.path.dirname(weights_path)
    classes = perf.c.shape[0]
    print('perf.c.shape[0]', classes)
    # 正常的num_classes应该是这个数据集num_classes-1，然而若是num_classes的话就表明有的像素点被预测为了背景，但是原先是背景的像素点不计入评价指标，这是因为设置了ignore_index
    save_path = os.path.join(save_dir, "classification_summary of {} in {} with rate of {}.txt").format(model, dataset, rate_trainset)
    with open(save_path, "w") as f:
        for i in range(classes):
            f.write(f"Class {i+1}:\n")
            for j in range(classes):
                if i != j and perf.c[i, j] > 0:
                    f.write(f"  {perf.c[i, j]} samples were misclassified as class {j+1}\n")
            f.write("\n")
    print(f"总结已保存到 {save_path} 文件中")
    #######################################################################
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 假设 perf.c 是一个 numpy 数组，包含类别之间的像素点个数
    # classes 是类别数
    classes = perf.c.shape[0]  # 获取类别个数
    confusion_matrix = perf.c  # 混淆矩阵（像素点个数）

    # 类别名称（根据具体任务替换）
    class_names = [f"{i+1}" for i in range(classes)]  # 假设类别名称为 Class 0, Class 1, ...

    # 设置图像大小
    plt.figure(figsize=(14, 12))
    import matplotlib.colors as mcolors

    # 创建单色颜色图
    single_color = mcolors.ListedColormap(['#FFFACD'])
    ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cbar=False, cmap=single_color, linewidths=1)
    rows, cols = confusion_matrix.shape
    for i in range(min(rows, cols)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='gold', lw=0))  # 对角线加深颜色
    # 添加轴标签和标题
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title("Confusion Matrix: Pixel Counts per Class", fontsize=16)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # 调整标签角度和布局
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "classification_summary of {} in {} with rate of {}.png").format(model, dataset, rate_trainset))
    #######################################################################

    print('Statistics on Test set:')
    if classes == args.num_class-1:
        Overall_accuracy = scores[0] * 100
        Average_Accuracy = scores[1] * 100
        Mean_IOU = scores[2] * 100
        Mean_DICE_score = scores[3] * 100
        FWIOU = scores[-1]
        print('Overall_accuracy {:.2f}%\nAverage_Accuracy {:.2f}%\nMean_IOU {:.2f}%\
                  \nMean_DICE_score {:.2f}'.format(Overall_accuracy, Average_Accuracy, Mean_IOU, Mean_DICE_score))
        print('FWIoU {:.2f}%'.format(FWIOU * 100))
        # print("***add part------**_IOU*f1score*_precision*_recall***************")
        print(
            '_IOU {} %\n_f1Score {} %\n_precision {}\n_recall {}'.format(scores[4] * 100, scores[5] * 100, scores[6] * 100,
                                                                         scores[7] * 100))
        print('test zongshu', add)
        whole_time = sum(whole_time)
        print('max_idx', max_idx)
        print('avg time per a small image:', whole_time / max_idx)
        print(f'FLOPs: {flops / 1e9:.2f} GFLOPs')  # 转换为 GFLOPs
        print(f'Parameters: {params / 1e6:.2f} M')

        return '{:.2f}'.format(Overall_accuracy), '{:.2f}'.format(Average_Accuracy), '{:.2f}'.format(Mean_IOU), '{:.2f}'.format(Mean_DICE_score), '{:.2f}'.format(FWIOU * 100)
    elif classes == args.num_class:
        FWIOU = scores[-1]
        if args.hsi_c == 'rad':
            Overall_accuracy = scores[0] * 100
            Average_Accuracy = scores[6][:5].sum() / (classes - 1) * 100
            # Average_Accuracy = scores[1] * 100
            Mean_IOU = scores[4][:5].sum() / (classes - 1) * 100
            # Mean_IOU = scores[2] * 100
            # Mean_DICE_score = scores[3] * 100
            Mean_DICE_score = scores[5][:5].sum() / (classes - 1) * 100
        else:
            Overall_accuracy = scores[0] * 100
            Average_Accuracy = scores[6][1:].sum()/(classes-1) * 100
            # Average_Accuracy = scores[1] * 100
            Mean_IOU = scores[4][1:].sum()/(classes-1) * 100
            # Mean_IOU = scores[2] * 100
            # Mean_DICE_score = scores[3] * 100
            Mean_DICE_score = scores[5][1:].sum()/(classes-1) * 100
        print('Overall_accuracy {:.2f}%\nAverage_Accuracy {:.2f}%\nMean_IOU {:.2f}%\
                  \nMean_DICE_score {:.2f}'.format(Overall_accuracy, Average_Accuracy, Mean_IOU, Mean_DICE_score))
        print('FWIoU {:.2f}%'.format(FWIOU * 100))
        # print("***add part------**_IOU*f1score*_precision*_recall***************")
        print(
            '_IOU {} %\n_f1Score {} %\n_precision {}\n_recall {}'.format(scores[4] * 100, scores[5] * 100,
                                                                         scores[6] * 100,
                                                                         scores[7] * 100))
        print('test zongshu', add)
        whole_time = sum(whole_time)
        print('max_idx', max_idx)
        print('avg time per a small image:', whole_time / max_idx)
        print(f'FLOPs: {flops / 1e9:.2f} GFLOPs')  # 转换为 GFLOPs
        print(f'Parameters: {params / 1e6:.2f} M')

        return '{:.2f}'.format(Overall_accuracy), '{:.2f}'.format(Average_Accuracy), '{:.2f}'.format(
            Mean_IOU), '{:.2f}'.format(Mean_DICE_score), '{:.2f}'.format(FWIOU * 100)

if __name__ == "__main__":
    # Overall_accuracy, Average_Accuracy, Mean_IOU, Mean_DICE_score, FWIOU = test_model('//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestoa_//rad// final use dilated_branches.pt_630.pt')
    Overall_accuracy, Average_Accuracy, Mean_IOU, Mean_DICE_score, FWIOU = test_model('//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestoa_//rad//lskandreparamed with LOSS.pt')

