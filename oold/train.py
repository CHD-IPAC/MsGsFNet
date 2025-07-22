import torch
import torch.utils.data as data
import numpy as np
from torch import optim
import torch.nn as nn
from torchvision import transforms
import math
import random
import os
import datetime
from helpers.augmentations import RandomHorizontallyFlip, RandomVerticallyFlip, \
    RandomTranspose, Compose

from helpers.utils import AverageMeter, Metrics, parse_args, LongKouLoader, HongHuLoader, HanChuanLoader, QingyunLoader
from helpers.lossfunctions import cross_entropy2d
from helpers.lossfunctions import SoftIoULoss

from networks.model_utils import init_weights, load_weights
import argparse
from einops import rearrange, reduce
import datetime
from datetime import datetime
from timm.models.vision_transformer import _cfg
import glob
import torch.nn.functional as F

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 设置随机数种子  180
setup_seed(140)                                 # 140 for HH and HC, 340 for QY,  80 for LK


def CenterClipper(CosineDistance_criterion):
    CosineDistance_criterion.weightcenters.data.clamp_(0, 1)


def train(epoch=0):
    global trainloss
    trainloss2 = AverageMeter()

    print('\nTrain Epoch: %d' % epoch)

    net.train()

    running_loss = 0.0

    # for idx, (hsi_ip, labels) in enumerate(trainloader, 0):
    criterion_name = str(criterion)
    print('这次使用的loss函数是:', criterion_name)
    for idx, batch in enumerate(trainloader, 0):
        hsi_ip = batch['hsi']
        labels = batch['label']

        N = hsi_ip.size(0)
        optimizer.zero_grad()
        outputs, att = net(hsi_ip.to(device))
        if args.network_arch == 'MsGsFNet':
            labels1 = rearrange(labels, 'b s c -> b (s c)', b=labels.shape[0])
            loss = criterion(outputs, labels.to(device))  # 求的是一个batch的平均loss,本身labels的shape是[32, 32, 32]，只有一个通道。但outputs是[32, 23, 32, 32]，
            loss2 = CosineDistance_criterion(att, labels1)
            loss += args.mu * loss2
            optimizer.zero_grad()
            optimizer_cd.zero_grad()
            loss.backward()
            for param in CosineDistance_criterion.parameters():
                param.grad.data *= (1. / args.mu)
                optimizer_cd.step()
            CenterClipper(CosineDistance_criterion)
            optimizer.step()
        else:
            loss = criterion(outputs, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        trainloss2.update(loss.item(), N)

        if (idx + 1) % 5 == 0:
            print('[Epoch %d, Batch %5d] 每一个batch的平均loss是: %.3f' % (epoch + 1, idx + 1, running_loss / 5))
            running_loss = 0.0

    trainloss.append(trainloss2.avg)  # 把每一个batch的平均loss加入列表


def val(epoch=0):
    global valloss
    valloss2 = AverageMeter()
    truth = []
    pred = []

    print('\nVal Epoch: %d' % epoch)

    net.eval()

    valloss_fx = 0.0

    with torch.no_grad():
        for idx, batch in enumerate(valloader, 0):
            hsi_ip = batch['hsi'].to(device=device)
            labels = batch['label'].to(device=device)
            N = hsi_ip.size(0)
            outputs, att = net(hsi_ip.to(device))
            if args.network_arch == 'MsGsFNet':
                labels1 = rearrange(labels, 'b s c -> b (s c)', b=labels.shape[0])
                loss = criterion(outputs, labels.to(device))  # 求的是一个batch的平均loss
                loss2 = CosineDistance_criterion(att, labels1)
                loss += args.mu * loss2
            else:
                loss = criterion(outputs, labels.to(device))

            valloss_fx += loss.item()

            valloss2.update(loss.item(), N)
            truth = np.append(truth, labels.cpu().numpy())                   # print(outputs.shape)   torch.Size([32, 23, 32, 32])---honghu
            pred = np.append(pred, outputs.max(1)[1].cpu().numpy())          # outputs.max(1)[1] 用于从模型的输出中提取32个样本所有通道预测的类别概率最大值的索引，(1)表示第一个维度，所以pred的shape也应是一个个[32,32,32]

    print('VAL: %d loss: %.3f' % (epoch + 1, valloss_fx / (idx + 1)))
    valloss.append(valloss2.avg)
    perf = Metrics(ignore_index=0)
    return perf(truth, pred), (valloss_fx / (idx + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AeroRIT baseline evalutions')

    ### 0. Config file?
    parser.add_argument('--config-file', default=None, help='Path to configuration file')
    ##################################################################################
    ### 1. Data Loading
    # parser.add_argument('--hsi_c', default='HanChuan', help='Load HSI Radiance or Reflectance data?')
    parser.add_argument('--hsi_c', default='HongHu', help='Load HSI Radiance or Reflectance data?')
    # parser.add_argument('--hsi_c', default='LongKou', help='Load HSI Radiance or Reflectance data?')
    # parser.add_argument('--hsi_c', default='Qingyun', help='Load HSI Radiance or Reflectance data?')

    # parser.add_argument('--rate_of_trainset', default='45')
    # parser.add_argument('--rate_of_trainset', default='40')
    # parser.add_argument('--rate_of_trainset', default='35')
    # parser.add_argument('--rate_of_trainset', default='30')
    parser.add_argument('--rate_of_trainset', default='25')
    # parser.add_argument('--rate_of_trainset', default='20')
    # parser.add_argument('--rate_of_trainset', default='15')
    # parser.add_argument('--rate_of_trainset', default='10')
    # parser.add_argument('--rate_of_trainset', default='5')
    # parser.add_argument('--rate_of_trainset', default='4')
    # parser.add_argument('--rate_of_trainset', default='3')
    # parser.add_argument('--rate_of_trainset', default='2')
    # parser.add_argument('--rate_of_trainset', default='1')
    # parser.add_argument('--rate_of_trainset', default='0.5')

    # parser.add_argument('--patchsize', default='4')
    # parser.add_argument('--patchsize', default='8')
    # parser.add_argument('--patchsize', default='16')
    parser.add_argument('--patchsize', default='32')

    if parse_args(parser).hsi_c =='HanChuan':
        parser.add_argument('--bands', default=274, help='Which bands category to load \
                                - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type=int)
        parser.add_argument('--num_class', default=17, help='The number of categories of the data', type=int)
        parser.add_argument('--inner_nc', default=64, type=int)
    elif parse_args(parser).hsi_c =='HongHu':
        parser.add_argument('--bands', default=270, help='Which bands category to load \
                                - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type=int)
        parser.add_argument('--num_class', default=23, help='The number of categories of the data', type=int)
        parser.add_argument('--inner_nc', default=64, type=int)
    elif parse_args(parser).hsi_c =='LongKou':
        parser.add_argument('--bands', default=270, help='Which bands category to load \
                                - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type=int)
        parser.add_argument('--num_class', default=10, help='The number of categories of the data', type=int)
        parser.add_argument('--inner_nc', default=64, type=int)
    elif parse_args(parser).hsi_c =='rad':
        parser.add_argument('--bands', default=51, help='Which bands category to load \
                                - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type=int)
        parser.add_argument('--num_class', default=6, help='The number of categories of the data', type=int)
        parser.add_argument('--inner_nc', default=256, type=int)
    elif parse_args(parser).hsi_c =='Tangdaowan':
        parser.add_argument('--bands', default=176, help='Which bands category to load \
                                - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type=int)
        parser.add_argument('--num_class', default=19, help='The number of categories of the data', type=int)
        parser.add_argument('--inner_nc', default=256, type=int)
    elif parse_args(parser).hsi_c =='Qingyun':
        parser.add_argument('--bands', default=176, help='Which bands category to load \
                                - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type=int)
        parser.add_argument('--num_class', default=7, help='The number of categories of the data', type=int)
        parser.add_argument('--inner_nc', default=64, type=int)
    elif parse_args(parser).hsi_c =='Pingan':
        parser.add_argument('--bands', default=176, help='Which bands category to load \
                                - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type=int)
        parser.add_argument('--num_class', default=11, help='The number of categories of the data', type=int)
        parser.add_argument('--inner_nc', default=256, type=int)

    ##################################################################################

    ### 2.Network selections
    parser.add_argument('--network_arch', default='MsGsFNet', help='Network a  rchitecture?')

    parser.add_argument('--network_bestoa_weights_path',
                        default='//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestoa_//{}//{} of patch{} in rate of {}.pt'.format(parse_args(parser).hsi_c, parse_args(parser).network_arch, parse_args(parser).patchsize, parse_args(parser).rate_of_trainset),
                        help='Path to Saved Network weights')
    parser.add_argument('--network_bestMiou_weights_path',
                        default='//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestMiou_//{}//{} of patch{} in rate of {}.pt'.format(parse_args(parser).hsi_c, parse_args(parser).network_arch, parse_args(parser).patchsize, parse_args(parser).rate_of_trainset),
                        help='Path to Saved Network weights')
    parser.add_argument('--network_bestaa_weights_path',
                        default='//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestaa_//{}//{} of patch{} in rate of {}.pt'.format(parse_args(parser).hsi_c, parse_args(parser).network_arch, parse_args(parser).patchsize, parse_args(parser).rate_of_trainset),
                        help='Path to Saved Network weights')
    parser.add_argument('--network_bestmds_weights_path',
                        default='//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestmds_//{}//{} of patch{} in rate of {}.pt'.format(parse_args(parser).hsi_c, parse_args(parser).network_arch, parse_args(parser).patchsize, parse_args(parser).rate_of_trainset),
                        help='Path to Saved Network weights')

    ### 3.Use GPU or not
    parser.add_argument('--use_cuda', action='store_true', default=True, help='use GPUs?')

    ### 4.Hyperparameters
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images sampled per minibatch?')
    parser.add_argument('--init_weights', default='kaiming', help="Choose from: 'normal', 'xavier', 'kaiming'")
    parser.add_argument('--learning_rate', default=1e-3, type=int,
                        help='Initial learning rate for training the network?')  # 1e-4
    parser.add_argument('--epochs', default=250, type=int, help='Maximum number of epochs?')

    ### Pretrained representation present?
    parser.add_argument('--pretrained_weights', default=None, help='Path to pretrained weights for network')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay for swl (default: 1e-4)')
    parser.add_argument('--mu', type=float, default=90, help='weight of category consistency loss')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--T_max', default=int(20), type=float, help='T_max')
    parser.add_argument('--Adam_min', default=1e-5, type=float, help='Adam_min')
    args = parse_args(parser)
    print(args)

    if args.use_cuda or torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # 以下augs_tx 和 tx 在各个dataloder中重新定义，无作用
    augs_tx = None

    tx = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.bands == 3 or args.bands == 4 or args.bands == 6:
        hsi_mode = '{}b'.format(args.bands)
    elif args.bands == 31:
        hsi_mode = 'visible'
    else:
        hsi_mode = 'all'

    num_class = args.num_class

    if args.hsi_c == 'Qingyun':
        trainset = QingyunLoader(set_type='NEW_seed10_{}_train_aug'.format(args.rate_of_trainset), size=args.patchsize, hsi_mode='all', transforms=tx,
                                 augs=augs_tx)  # first_train second_train third_train fourth_train fifth_train
        valset = QingyunLoader(set_type='NEW_seed10_{}_train_aug'.format(args.rate_of_trainset), size=args.patchsize, hsi_mode='all', transforms=tx)
        ## first_val second_val third_val fourth_val fifth_val
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        weights = [1.0, 7.45, 10.65, 149.65, 152.88, 9.42, 8.14]
        weights = torch.FloatTensor(weights)
        criterion = cross_entropy2d(reduction='mean', weight=weights.cuda(), ignore_index=0)

    elif args.hsi_c == 'LongKou':
        trainset = LongKouLoader(set_type='NEW_seed10_{}_train_aug'.format(args.rate_of_trainset), size=args.patchsize, hsi_mode='all', transforms=tx,
                                 augs=augs_tx)  # first_train second_train third_train fourth_train fifth_train
        valset = LongKouLoader(set_type='NEW_seed10_{}_train_aug'.format(args.rate_of_trainset), size=args.patchsize, hsi_mode='all', transforms=tx)
        ## first_val second_val third_val fourth_val fifth_val
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        weights = [1.0, 8.44, 31.93, 74.69, 3.25, 37.89, 11.07, 3.07, 19.35, 28.07]
        weights = torch.FloatTensor(weights)
        criterion = cross_entropy2d(reduction='mean', weight=weights.cuda(), ignore_index=0)

    elif args.hsi_c == 'HongHu':
        trainset = HongHuLoader(set_type='NEW_seed10_{}_train_aug'.format(args.rate_of_trainset), size=args.patchsize, hsi_mode='all', transforms=tx,
                                    augs=augs_tx)  # first_train second_train third_train fourth_train fifth_train
        valset = HongHuLoader(set_type='NEW_seed10_{}_train_aug'.format(args.rate_of_trainset), size=args.patchsize, hsi_mode='all', transforms=tx)
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)   # True=0.65，Flase=0.66
        weights = [1.0, 35.01, 100.92, 16.13, 3.13, 64, 7.28, 11.39, 102.67, 25.68, 26.87, 32.15, 32.21, 13.62, 56.37,
                   235.48, 53.74, 116.68, 59.11, 43.87, 65.95, 300.89, 98.4, ]
        weights = torch.FloatTensor(weights)
        criterion = cross_entropy2d(reduction='mean', weight=weights.cuda(), ignore_index=0)

    elif args.hsi_c == 'HanChuan':
        trainset = HanChuanLoader(set_type='NEW_seed10_{}_train_aug'.format(args.rate_of_trainset), size=args.patchsize, hsi_mode='all', transforms=tx,
                                  augs=augs_tx)  ## first_train second_train third_train fourth_train fifth_train
        valset = HanChuanLoader(set_type='NEW_seed10_{}_train_aug'.format(args.rate_of_trainset), size=args.patchsize, hsi_mode='all', transforms=tx)
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        weights = [1.0, 5.67, 7.66, 24.01, 36.91, 135.91, 30.98, 34.01, 10.92, 20.27, 19.37, 12.18, 28.55, 23.88, 8.68,
                   88.01, 3.63]
        weights = torch.FloatTensor(weights)
        criterion = cross_entropy2d(reduction='mean', weight=weights.cuda(), ignore_index=0)

    if args.network_arch == 'MsGsFNet':
        from networks.MsGsFNet import MsGsFNet

        net = MsGsFNet(args.bands, num_class, False, args.inner_nc)

    elif args.network_arch == 'MsGsFNet_baseline':
        from networks.MsGsFNet_baseline import MsGsFNet

        net = MsGsFNet(args.bands, num_class, False, args.inner_nc)

    else:
        raise NotImplementedError('required parameter not found in dictionary')

    init_weights(net, init_type=args.init_weights)
    if args.pretrained_weights is not None:
        load_weights(net, args.pretrained_weights)
        print('Completed loading pretrained network weights')

    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.Adam_min)

    if args.network_arch == 'MsGsFNet':

       class CosineDistanceLoss(nn.Module):
           def __init__(self, num_classes, embedding_size):
               super(CosineDistanceLoss, self).__init__()
               self.num_classes = num_classes
               self.embedding_size = embedding_size
               self.weightcenters = nn.Parameter(torch.normal(0, 1, (num_classes, embedding_size)))

           def forward(self, x, labels):  # x是band weights，labels是targets
               if len(x.size()) == 1:
                   x = x.unsqueeze(0)
               batch_size = x.size(0)
               ####################################################
               x_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # x的L2范数, 形状为(batch_size, 1)
               weightcenters_norm = torch.norm(self.weightcenters, p=2, dim=1,
                                               keepdim=True)  # weightcenters的L2范数, 形状为(num_classes, 1)
               x_norm_expand = x_norm.squeeze(dim=1).squeeze(dim=1).expand(batch_size, self.num_classes).t()
               weightcenters_norm_expand = weightcenters_norm.expand(self.num_classes, batch_size).t()
               x = x.view(batch_size, -1)
               cosine_similarity = torch.mm(x, self.weightcenters.t()) / (
                           x_norm_expand.t() * weightcenters_norm_expand + 1e-10)
               cosine_distance = 1 - cosine_similarity
               dist_metric = cosine_distance
               ####################################################
               labels = labels.to(dist_metric.device)
               selected_values = torch.gather(dist_metric, 1, labels)
               final_sum_per_sample = selected_values.sum(dim=1)
               loss = final_sum_per_sample.clamp(1e-12, 1e+12).sum() / batch_size
               return loss

    trainloss = []
    valloss = []
    bestmiou = 0
    bestoa = 0
    bestaa = 0
    bestmds = 0
    bestmIOUepoch = 0
    bestOAepoch = 0
    bestaaepoch = 0
    bestmdsepoch = 0
    path = args.network_bestMiou_weights_path
    path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)
    ###################################################
    if args.network_arch == 'MsGsFNet':
        embedding_size = 256
        CosineDistance_criterion = CosineDistanceLoss(num_classes=args.num_class, embedding_size=embedding_size)
        CosineDistance_criterion = CosineDistance_criterion.cuda()
        categotyconsistencyloss_paras_group = [{'params': CosineDistance_criterion.parameters(), 'weight_decay': args.weight_decay}]
        optimizer_cd = torch.optim.SGD(categotyconsistencyloss_paras_group, args.lr,
                                       momentum=args.momentum, nesterov=True)

    for epoch in range(args.epochs):

        train(epoch)
        (Overall_accuracy, Average_Accuracy, Mean_IOU, Mean_DICE_score, _, _, _, _, _), loss = val(epoch)
        print('Overall acc  = {:.3f}, MPCA = {:.3f}, mIOU = {:.3f}'.format(Overall_accuracy, Average_Accuracy, Mean_IOU))

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print("The weights occur time:", current_time)
        print('The best mIOU is', bestmiou)
        print('The best mIOU epoch is', bestmIOUepoch)
        print('The best oa is', bestoa)
        print('The best oa epoch is', bestOAepoch)
        print('The best aa is', bestaa)
        print('The best aa epoch is', bestaaepoch)
        print('The best mds is', bestmds)
        print('The best mds epoch is', bestmdsepoch)

        if Overall_accuracy > bestoa:
           bestoa = Overall_accuracy
           bestOAepoch = epoch
           torch.save(net.state_dict(), args.network_bestoa_weights_path)
        if Mean_IOU > bestmiou:
            bestmiou = Mean_IOU
            bestmIOUepoch = epoch
            torch.save(net.state_dict(), args.network_bestMiou_weights_path)
        if Average_Accuracy > bestaa:
            bestaa = Average_Accuracy
            bestaaepoch = epoch
            torch.save(net.state_dict(), args.network_bestaa_weights_path)
        if Mean_DICE_score > bestmds:
            bestmds = Mean_DICE_score
            bestmdsepoch = epoch
            torch.save(net.state_dict(), args.network_bestmds_weights_path)
        scheduler.step()  # 调整学习率
    print(datetime.now())
