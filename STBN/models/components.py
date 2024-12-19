import torch.nn as nn
import torch
from models.init import init_fn
import functools
import sys

sys.path.append('..')


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        output = self.conv2(self.relu(self.conv1(x)))
        output = torch.add(output, x)
        return output


class ResBlocks(nn.Module):
    def __init__(self, input_channels, num_resblocks, num_channels):
        super(ResBlocks, self).__init__()
        self.input_channels = input_channels
        self.first_conv = nn.Conv2d(in_channels=self.input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        modules = []
        for _ in range(num_resblocks):
            modules.append(ResBlock(in_channels=num_channels, mid_channels=num_channels, out_channels=num_channels))
        self.resblocks = nn.Sequential(*modules)

        fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        self.apply(fn)

    def forward(self, h):
        shallow_feature = self.first_conv(h)
        new_h = self.resblocks(shallow_feature)
        return new_h


class ResConv(nn.Module):
    def __init__(self, input_channels, mid_channels, out_channels, num_resblocks):
        super(ResConv, self).__init__()
        self.input_channels = input_channels
        self.first_conv = nn.Conv2d(in_channels=self.input_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

        modules = []
        for _ in range(num_resblocks):
            modules.append(ResBlock(in_channels=mid_channels, mid_channels=mid_channels, out_channels=out_channels))
        self.resblocks = nn.Sequential(*modules)

    def forward(self, h):
        shallow_feature = self.first_conv(h)
        new_h = self.resblocks(shallow_feature)
        return new_h


class D(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(D, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.convs = nn.Sequential(*layers)

        fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        self.apply(fn)

    def forward(self, x):
        x = self.convs(x)
        return x


class D_dilate(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_res=9):
        super(D_dilate, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers += [DCl(2, mid_channels) for _ in range(num_res)]
        # layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.convs = nn.Sequential(*layers)

        # fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        # self.apply(fn)

    def forward(self, x):
        x = self.convs(x)
        return x


class D_dilate_u(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_res=9):
        super(D_dilate_u, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers += [DCl(2, mid_channels) for _ in range(num_res)]
        # layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.convs = nn.Sequential(*layers)

        fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        self.apply(fn)

    def forward(self, x):
        x = self.convs(x)
        return x


class D_fin(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_res=9):
        super(D_fin, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers += [DCl(2, mid_channels) for _ in range(num_res)]
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels//2, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=mid_channels//2, out_channels=mid_channels//4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=mid_channels//4, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.convs = nn.Sequential(*layers)

        # fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        # self.apply(fn)

    def forward(self, x):
        x = self.convs(x)
        return x


class D_dial_u(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_res=9):
        super(D_dial_u, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers += [DCl(2, mid_channels) for _ in range(num_res)]
        # layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.convs = nn.Sequential(*layers)

        fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        self.apply(fn)

    def forward(self, x):
        x = self.convs(x)
        return x
