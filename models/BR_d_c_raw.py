from tqdm import tqdm
from models.SRFE.SRFE import SRFE
from models.DBSN.DBSNl import DBSNl
from models.components import D_dilate, D_fin, ResBlocks, D
from pytorch_pwc.extract_flow import extract_flow_torch, get_flow_2frames
from pytorch_pwc.pwc import PWCNet
import torch
import torch.nn as nn
from utils.raw import demosaic
from utils.warp import warp
from torchvision.utils import save_image
from torchvision.utils import flow_to_image


class BR_d_c_raw(nn.Module):
    def __init__(self, img_channels=1, num_resblocks=5, num_channels=64):
        super(BR_d_c_raw, self).__init__()
        self.num_channels = num_channels
        self.pwcnet = PWCNet()
        self.forward_rnn = DBSNl(in_ch=img_channels + num_channels, out_ch=num_channels, base_ch=num_channels, num_module=num_resblocks)
        self.backward_rnn = DBSNl(in_ch=img_channels + num_channels, out_ch=num_channels, base_ch=num_channels, num_module=num_resblocks)
        # self.forward_rnn = LGBPN(in_ch=img_channels + num_channels, out_ch=num_channels, base_ch=num_channels, num_module=9, br2_blc=3)
        # self.backward_rnn = LGBPN(in_ch=img_channels + num_channels, out_ch=num_channels, base_ch=num_channels, num_module=9, br2_blc=4)
        self.concat = D_dilate(in_channels=num_channels*2, mid_channels=num_channels*2, out_channels=num_channels, num_res=num_resblocks)
        # self.d = D_dial(in_channels=num_channels * 2, mid_channels=num_channels * 2, out_channels=img_channels+6,num_res=4)

        self.d = SRFE(in_channels=num_channels)
        self.d2 = D_dilate(in_channels=num_channels, mid_channels=num_channels, out_channels=img_channels, num_res=num_resblocks//2)

    def trainable_parameters(self):
        return [{'params': self.d2.parameters()}, {'params': self.forward_rnn.parameters()}, {'params': self.concat.parameters()},
                {'params': self.backward_rnn.parameters()}, {'params': self.d.parameters()}]  # , 'lr': 3e-5

    def forward(self,  seqn):
        if self.training:
            return self.forward_training(seqn)
        else:
            return self.forward_testing(seqn)

    def forward_testing(self, seqn):
        if self.training:
            feature_device = torch.device('cuda')
        else:
            feature_device = torch.device('cpu')
        N, T, C, H, W = seqn.shape
        forward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        backward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        seqdn = torch.empty(N, T, C, H, W, device='cuda')
        srgb_seqn = demosaic(seqn)
        # extract forward features
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        forward_h = self.forward_rnn(torch.cat((seqn[:, 0], init_forward_h), dim=1))
        forward_hs[:, 0] = forward_h.to(feature_device)
        for i in range(1, T):
            flow = extract_flow_torch(self.pwcnet, srgb_seqn[:, i], srgb_seqn[:, i-1])
            aligned_forward_h, _ = warp(forward_h, flow)
            forward_h = self.forward_rnn(torch.cat((seqn[:, i], aligned_forward_h), dim=1))
            forward_h = self.concat(torch.cat((forward_h, aligned_forward_h), dim=1))
            forward_hs[:, i] = forward_h.to(feature_device)

        # extract backward features
        init_backward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        backward_h = self.backward_rnn(torch.cat((seqn[:, -1], init_backward_h), dim=1))
        backward_hs[:, -1] = backward_h.to(feature_device)
        for i in range(2, T+1):
            flow = extract_flow_torch(self.pwcnet, srgb_seqn[:, T-i], srgb_seqn[:, T-i+1])
            aligned_backward_h, _ = warp(backward_h, flow)
            backward_h = self.backward_rnn(torch.cat((seqn[:, T-i], aligned_backward_h), dim=1))
            backward_h = self.concat(torch.cat((backward_h, aligned_backward_h), dim=1))

            backward_hs[:, T-i] = backward_h.to(feature_device)

        # generate results
        iterate = range(T)  # if self.training else tqdm(range(T))
        for i in iterate:
            # seqdn[:, i] = self.d(torch.cat((forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device)), dim=1))
            temp = self.d(forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device))

            seqdn[:, i] = self.d2(temp)

        return seqdn

    def forward_training(self, seqn):
        if self.training:
            feature_device = torch.device('cuda')
        else:
            feature_device = torch.device('cpu')
        N, T, C, H, W = seqn.shape
        forward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        backward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        seqdn = torch.empty(N, T, C, H, W, device='cuda')
        srgb_seqn = demosaic(seqn)
        flows_backward, flows_forward = get_flow_2frames(self.pwcnet, srgb_seqn)

        # extract forward features
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        forward_h = self.forward_rnn(torch.cat((seqn[:, 0], init_forward_h), dim=1))
        forward_hs[:, 0] = forward_h.to(feature_device)
        for i in range(1, T):
            # flow = extract_flow_torch(self.pwcnet, srgb_seqn[:, i], srgb_seqn[:, i-1])
            aligned_forward_h, _ = warp(forward_h,  flows_forward[:, i-1].cuda())
            forward_h = self.forward_rnn(torch.cat((seqn[:, i], aligned_forward_h), dim=1))
            forward_h = self.concat(torch.cat((forward_h, aligned_forward_h), dim=1))
            forward_hs[:, i] = forward_h.to(feature_device)

        # extract backward features
        init_backward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        backward_h = self.backward_rnn(torch.cat((seqn[:, -1], init_backward_h), dim=1))
        backward_hs[:, -1] = backward_h.to(feature_device)
        for i in range(2, T+1):
            # flow = extract_flow_torch(self.pwcnet, srgb_seqn[:, T-i], srgb_seqn[:, T-i+1])
            aligned_backward_h, _ = warp(backward_h,  flows_backward[:, T-i].cuda())
            backward_h = self.backward_rnn(torch.cat((seqn[:, T-i],  aligned_backward_h), dim=1))
            backward_h = self.concat(torch.cat((backward_h, aligned_backward_h), dim=1))

            backward_hs[:, T-i] = backward_h.to(feature_device)

        # generate results
        # iterate = range(T) if self.training else tqdm(range(T))
        for i in range(T):
            # seqdn[:, i] = self.d(torch.cat((forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device)), dim=1))
            temp = self.d(forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device))

            seqdn[:, i] = self.d2(temp)

        return seqdn
