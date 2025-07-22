import torch
import torch.nn as nn
from torch.autograd import Function
import functools
import math
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.nn.modules.utils import _pair


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.S = getattr(args, 'MD_S', 1)
        self.D = getattr(args, 'MD_D', 512)

        # self.R = getattr(args, 'MD_R', 5)  # 5 for LK and QY.
        # self.R = getattr(args, 'MD_R', 32)   #  32 for HC.
        self.R = getattr(args, 'MD_R', 16)  # 16 for HH.

        self.train_steps = getattr(args, 'TRAIN_STEPS', 6)
        self.eval_steps = getattr(args, 'EVAL_STEPS', 7)

        self.inv_t = getattr(args, 'INV_T', 100)
        self.eta = getattr(args, 'ETA', 0.9)

        self.rand_init = getattr(args, 'RAND_INIT', True)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, S = x.shape

        D = S
        N = C // self.S
        x = x.view(B * self.S, N, D).transpose(1, 2)

        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)  # 构造后的bases = torch.rand((B*self.S,D,self.R))
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        coef = self.compute_coef(x, bases, coef)

        x = torch.bmm(bases, coef.transpose(1, 2))

        x = x.transpose(1, 2).view(B, C, S)

        # if not self.rand_init or return_bases:
        #     return x, bases
        # else:
        return x

    @torch.no_grad()
    def online_update(self, bases):
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class NMF2D(_MatrixDecomposition2DBase):   # 子类没有forward时，会借用父类的forward
    def __init__(self, args):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)
        # print(coef)
        return coef


def get_hams(key):
    hams = {
        'NMF': NMF2D}

    assert key in hams

    return hams[key]


class Hamburger(nn.Module):
    def __init__(self, in_c, args=None):
        super().__init__()
        C = in_c
        self.norm = nn.BatchNorm1d(C)
        self.lower_bread = nn.Sequential(nn.Conv1d(C, C, 1), nn.ReLU(inplace=True))

        HAM = NMF2D
        self.NVR = HAM(args)
        self.NVR.D = in_c

        self.upper_bread = nn.Conv1d(C, C, 1, bias=False)
        self.shortcut = nn.Sequential()

    def forward(self, x):  # x:  (b h w) c s
        # B, C, S, H, W = x.shape

        # ham_x = self.norm(x)
        x1 = x
        x1 = self.lower_bread(x1)
        x1 = self.NVR(x1)
        # ham_x = self.upper_bread(ham_x)
        out = F.relu(x + x1, inplace=True)
        return out

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)


class AMPooling(nn.Module):
    def __init__(self, num_channels=256):
        super(AMPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels // 4, out_channels=num_channels, kernel_size=1, bias=False)
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
        x_avg, x_max = self.avgpool(x), self.maxpool(x)
        avg_weights, max_weights = self.channel_attention(x_avg), self.channel_attention(x_max)
        band_weights = self.sigmoid(avg_weights + max_weights)
        return band_weights, band_weights*x


class VRAttention(nn.Module):

    def __init__(self, inplanes, stride=1, attention='1'):
        super(VRAttention, self).__init__()

        if inplanes < 64:
            planes = 64
            att_dim = 32
        else:
            planes = 256
            att_dim = 128
        self.AMPooling = AMPooling(planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes),
        )
        self.ham = Hamburger(planes)
        # self.spectral_ffn = nn.Sequential(
        #     nn.BatchNorm1d(planes),
        #     nn.Conv1d(planes, planes, kernel_size=1, stride=1,
        #               padding=0),
        #     nn.GELU(),
        #     nn.Conv1d(planes, planes, kernel_size=1, stride=1,
        #               padding=0),
        # )
        self.fc_adapt_channels1 = nn.Conv2d(
            planes, inplanes,
            kernel_size=1, groups=1, bias=True)

    def forward(self, x):
        # 增加维度
        x1 = x[:, None]
        b, s, c, h, w = x1.shape
        x2 = rearrange(x1, 'b s c h w -> (b s) c h w')  # b*s c h w
        x3 = self.conv1(x2)  # 改变通道为2的次方，对硬件友好
        x3 = self.conv3(x3)  # 深度卷积, 以上两个卷积层为空谱数据的融合，对空间数据也做以去噪

        h_, w_ = x3.shape[-2], x3.shape[-1]
        x3 = rearrange(x3, '(b s) c h w -> (b h w) c s', s=s)
        x3 = self.ham(x3)
        x3 = reduce(x3, '(b h w) c s -> b c h w', 'mean', b=b, h=h_, w=w_)
        out = x3
        att, out = self.AMPooling(out)
        out = self.fc_adapt_channels1(out)

        return out, att


if __name__ == '__main__':
    input_channels = 270
    H = 32
    W = H
    model = VRAttention(inplanes=input_channels)
    # model = ResBlock_CBAM(in_places=16, places=4)
    # print(model)
    input = torch.randn(32, input_channels, H, W)
    print('input', input.shape)
    out = model(input)
    print('output', out.shape)
