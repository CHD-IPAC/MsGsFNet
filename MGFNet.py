import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

class GSA(nn.Module):
    def __init__(self, inplanes, stride=1, attention='1'):
        super(GSA,self).__init__()
        if inplanes < 64:
            planes = 64
            att_dim = 32
        else:
            planes = 256
            att_dim = 128
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 =   nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.relu_normal = nn.ReLU(inplace=False)

        if attention in {'1','+','M','&'}:
            if planes > 64:
                DR_stride=1
            else:
                DR_stride=2
            self.ch_dim = att_dim
            self.conv_for_DR = nn.Conv2d(
                 planes, self.ch_dim,
                 kernel_size=1,stride=DR_stride, bias=True)
            self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
            self.row_bn = nn.BatchNorm2d(self.ch_dim)

            self.row_conv_group = nn.Conv2d(
                 self.ch_dim, 4*self.ch_dim,
                 kernel_size=(self.ch_dim, 1),
                 groups = self.ch_dim, bias=True)
            self.fc_adapt_channels = nn.Conv2d(
                 4*self.ch_dim, inplanes,
                 kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()

        self.stride = stride
        self.attention = attention

    def chan_att(self, out):
        out = self.conv_for_DR(out)
        out = self.bn_for_DR(out)
        out = self.relu(out)
        out = CovpoolLayer(out)
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()
        out = self.row_bn(out)
        out = self.row_conv_group(out)
        out = self.fc_adapt_channels(out)
        out = self.sigmoid(out)
        return out

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.attention == '1':
            att = self.chan_att(out)
            out = residual * att
        return out

def feature_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True))

def SplitChannels(channels, kernel_num):
    split_channels = [channels//kernel_num for _ in range(kernel_num)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

class EMFF(nn.Module):
    def __init__(self, channels, kernel_num=4):
        super(EMFF, self).__init__()
        self.channels = channels
        self.kernel_num = kernel_num
        self.sp = SplitChannels(self.channels, self.kernel_num)
        dilations = [1, 2, 3, 4]
        self.conv1_1 = feature_branch(self.sp[0], self.sp[0], 3, dilation=dilations[0])
        self.conv1_2 = feature_branch(self.sp[1], self.sp[1], 3, dilation=dilations[1])
        self.conv1_3 = feature_branch(self.sp[2], self.sp[2], 3, dilation=dilations[2])
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(self.sp[3]),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_split = torch.split(x, self.sp, dim=1)
        x_1_1 = self.conv1_1(x_split[0])
        x_1_2 = self.conv1_2(x_split[1]+x_1_1)
        x_1_3 = self.conv1_3(x_split[2]+x_1_2)
        x_1_4 = F.interpolate(self.avg_pool(x_split[3]+x_1_3), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat([x_1_1,x_1_2,x_1_3,x_1_4], dim=1)
        return x

class GSSI(nn.Module):
    def __init__(self, in_channels_1, in_channels_2,out_channels):
        super(GSSI, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_1, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels_2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.conva = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,1, padding=0,bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,1,1,padding=0,bias=True),
            nn.Sigmoid(),
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
        self.project = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self,x1,x2):
        x1 = self.conv1(x1)
        x = self.conv2(x2)
        x2 = F.interpolate(x, size=(2 * x.shape[2], 2 * x.shape[3]), mode='bilinear',
                      align_corners=False)
        g1 = self.conva(x1)
        g2 = self.convb(x2)
        a_gff = (1 + g1) * x1 + (1 - g1) * (g2 * x2)
        b_gff = (1 + g2) * x2 + (1 - g2) *  (g1 * x1)
        gff_outs = torch.cat([a_gff, b_gff], dim=1)
        output = self.project(gff_outs)
        return output

class MGFNet(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(MGFNet, self).__init__()
        self.input_nc = input_nc
        self.GSA = GSA(inplanes=input_nc)
        filters = [64, 128, 256]
        self.output_nc = output_nc
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc, filters[0], kernel_size=3, stride=1, padding=1),
            norm_layer(filters[0], affine=True),
            nn.ReLU(True))
        self.GSSI_1 = GSSI(filters[0],filters[1],filters[0])
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3,stride=2, padding=1),
            norm_layer(filters[1], affine=True),
            nn.ReLU(True))
        self.GSSI_2 = GSSI(filters[1],filters[2],filters[1])
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3,stride=2, padding=1),
            norm_layer(filters[2], affine=True),
            nn.ReLU(True))
        self.EMFF = EMFF(filters[2])
        self.conv4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1),
            norm_layer(filters[1], affine=True),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0],kernel_size=3, padding=1),
            norm_layer(filters[0], affine=True),
            nn.ReLU(True)
        )
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
        input = self.GSA(input)
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x1_se = self.GSSI_1(x1, x2)
        x2_se = self.GSSI_2(x2, x3)
        x_block = self.EMFF(x3)
        x4_temp = F.interpolate((x_block + x3),scale_factor=2, mode='bilinear', align_corners=False)
        x4 = self.conv4(x4_temp)
        x5_temp = F.interpolate((x4 + x2_se), scale_factor=2, mode='bilinear', align_corners=False)
        x5 = self.conv5(x5_temp)
        output = self.lastconv(x5 + x1_se)
        return output
