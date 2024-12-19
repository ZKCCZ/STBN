import torch.nn as nn
import torch
from models.init import init_fn
import functools
import sys

# from models.DBSN.DBSNl_simam import simam_module

sys.path.append('..')
# from models.DBSN.DBSNl import simam_module


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


class ResBlocks_2(nn.Module):
    def __init__(self, input_channels, num_resblocks, num_channels, output_channels):
        super(ResBlocks_2, self).__init__()
        self.input_channels = input_channels
        self.first_conv = nn.Conv2d(in_channels=self.input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        modules = []
        for _ in range(num_resblocks):
            modules.append(ResBlock(in_channels=num_channels, mid_channels=num_channels, out_channels=num_channels))
        self.resblocks = nn.Sequential(*modules)
        self.last_conv = nn.Conv2d(in_channels=num_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False)

        fn = functools.partial(init_fn, init_type='kaiming_normal', init_bn_type='uniform', gain=0.2)
        self.apply(fn)

    def forward(self, h):
        shallow_feature = self.first_conv(h)
        new_h = self.resblocks(shallow_feature)
        new_h = self.last_conv(new_h)

        return new_h


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
