# This file uses code from the CAIN project by Myungsub Choi
# GitHub Repository: https://github.com/myungsub/CAIN
# Copyright (c) 2020 Myungsub Choi
# Licensed under the MIT License (https://opensource.org/licenses/MIT)
# See the original project for more details.

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class SRFE(nn.Module):
    def __init__(self, in_channels=3, depth=1):
        super(SRFE, self).__init__()
        self.padder_size = 4 ** depth
        self.encoder = Encoder(in_channels=in_channels, depth=depth, n_resgroups=2, n_resblocks=8)
        self.decoder = Decoder(depth=depth)

    def check_image_size(self, x1, x2):
        assert x1.shape == x2.shape
        _, _, h, w = x1.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x1 = F.pad(x1, (0, mod_pad_w, 0, mod_pad_h))
        x2 = F.pad(x2, (0, mod_pad_w, 0, mod_pad_h))
        return x1, x2

    def forward(self, x1, x2):
        _, _, H, W = x1.shape
        x1, x2 = self.check_image_size(x1, x2)

        feats = self.encoder(x1, x2)
        out = self.decoder(feats)

        return out[:, :, :H, :W]


class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(RCAB, self).__init__()
        stride = 2
        self.body = DCl(stride, in_feat)

    def forward(self, x):
        out = self.body(x)
        out = out + x
        return out


class fusion(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats):
        super(fusion, self).__init__()

        stride = 2
        self.headConv = nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding=stride, dilation=stride)

        self.body = nn.Sequential(*[ResidualGroup(RCAB, n_resblocks=n_resblocks, n_feat=n_feats) for _ in range(n_resgroups)])

        self.tailConv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=stride, dilation=stride),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1)
        )

    def forward(self, x0, x1):
        x = torch.cat([x0, x1], dim=1)
        x = self.headConv(x)
        res = self.body(x)
        out = self.tailConv(res)
        return out


class ResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat):
        super(ResidualGroup, self).__init__()
        modules_body = [Block(n_feat, n_feat) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()
        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride, groups=2)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3, n_resgroups=2, n_resblocks=5):
        super(Encoder, self).__init__()

        self.shuffler = Downsample(2)
        self.interpolate = fusion(n_resgroups, n_resblocks, in_channels * (4**depth))

    def forward(self, x1, x2):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)

        feats = self.interpolate(feats1, feats2)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        self.shuffler = Upsample(2)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class Downsample(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c (hd h) (wd w) -> b (c hd wd) h w', h=self.dilation**2, w=self.dilation**2)
        x = rearrange(x, 'b c (hn hh) (wn ww) -> b c (hn wn) hh ww', hh=self.dilation, ww=self.dilation)
        x = rearrange(x, 'b (c hd wd) cc hh ww-> b (c cc) (hd hh) (wd ww)', hd=H//(self.dilation**2), wd=W//(self.dilation**2))
        return x


class Upsample(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b (c cc) (hd hh) (wd ww) -> b (c hd wd) cc hh ww', cc=self.dilation**2, hh=self.dilation, ww=self.dilation)
        x = rearrange(x, 'b c (hn wn) hh ww -> b c (hn hh) (wn ww)', hn=self.dilation, wn=self.dilation)
        x = rearrange(x, 'b (c hd wd) h w -> b c (hd h) (wd w)', hd=H//self.dilation, wd=W//self.dilation)
        return x


def pixel_unshuffle(input, factor):
    """
    (n, c, h, w) ===> (n*factor^2, c, h/factor, w/factor)
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // factor
    out_width = in_width // factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, factor,
        out_width, factor)

    batch_size *= factor ** 2
    unshuffle_out = input_view.permute(0, 3, 5, 1, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


def pixel_shuffle(input, factor):
    """
    (n*factor^2, c, h/factor, w/factor) ===> (n, c, h, w)
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height * factor
    out_width = in_width * factor

    batch_size /= factor ** 2
    batch_size = int(batch_size)
    input_view = input.contiguous().view(
        batch_size, factor, factor, channels, in_height,
        in_width)

    unshuffle_out = input_view.permute(0, 3, 4, 1, 5, 2).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)
