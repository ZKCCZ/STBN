from tqdm import tqdm
from models.SRFE.SRFE import SRFE

from models.DBSN.DBSNl import DBSNl
from models.components import D_dilate, D_fin, ResBlocks, D
from pytorch_pwc.extract_flow import extract_flow_torch, get_flow_2frames, get_flow_2frames_train
from pytorch_pwc.pwc import PWCNet
import torch
import torch.nn as nn
from utils.warp import warp
import torch.nn.functional as F

pwcnet_fix = PWCNet().cuda()


class bdc_flow(nn.Module):
    def __init__(self, img_channels=3, num_resblocks=5, num_channels=64):
        super(bdc_flow, self).__init__()
        self.num_channels = num_channels
        self.pwcnet = PWCNet()

        self.forward_rnn = DBSNl(in_ch=img_channels + num_channels, out_ch=num_channels, base_ch=num_channels, num_module=num_resblocks)
        self.backward_rnn = DBSNl(in_ch=img_channels + num_channels, out_ch=num_channels, base_ch=num_channels, num_module=num_resblocks)
        self.concat = D_dilate(in_channels=num_channels*2, mid_channels=num_channels * 2, out_channels=num_channels, num_res=num_resblocks)

        self.d = SRFE(in_channels=num_channels)
        self.d2 = D_dilate(in_channels=num_channels, mid_channels=num_channels, out_channels=img_channels+6, num_res=num_resblocks//2)

    def trainable_parameters(self):
        return [{'params': self.d2.parameters()}, {'params': self.forward_rnn.parameters()}, {'params': self.concat.parameters()},
                {'params': self.backward_rnn.parameters()}, {'params': self.d.parameters()}]  # , 'lr': 3e-5

    def forward(self, seqn, noise_level_map=None):
        if self.training:
            return self.forward_training(seqn, noise_level_map)
        else:
            return self.forward_testing(seqn, noise_level_map)

    def forward_testing(self, seqn, noise_level_map=None):
        feature_device = torch.device('cpu')
        N, T, C, H, W = seqn.shape
        forward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        backward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        seqdn = torch.empty(N, T, C+6, H, W, device='cuda')

        # extract forward features
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        forward_h = self.forward_rnn(torch.cat((seqn[:, 0], init_forward_h), dim=1))
        forward_hs[:, 0] = forward_h.to(feature_device)
        for i in range(1, T):
            flow = extract_flow_torch(self.pwcnet, seqn[:, i], seqn[:, i-1])
            aligned_forward_h, _ = warp(forward_h, flow)
            forward_h = self.forward_rnn(torch.cat((seqn[:, i], aligned_forward_h), dim=1))
            forward_h = self.concat(torch.cat((forward_h, aligned_forward_h), dim=1))
            forward_hs[:, i] = forward_h.to(feature_device)

        # extract backward features
        init_backward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        backward_h = self.backward_rnn(torch.cat((seqn[:, -1], init_backward_h), dim=1))
        backward_hs[:, -1] = backward_h.to(feature_device)
        for i in range(2, T+1):
            flow = extract_flow_torch(self.pwcnet, seqn[:, T-i], seqn[:, T-i+1])
            aligned_backward_h, _ = warp(backward_h, flow)
            backward_h = self.backward_rnn(torch.cat((seqn[:, T-i],  aligned_backward_h), dim=1))
            backward_h = self.concat(torch.cat((backward_h, aligned_backward_h), dim=1))

            backward_hs[:, T-i] = backward_h.to(feature_device)

        # generate results
        for i in tqdm(range(T)):
            # seqdn[:, i] = self.d(torch.cat((forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device)), dim=1))
            temp = self.d(forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device))
            seqdn[:, i] = self.d2(temp)

        return seqdn, None

    def forward_training(self, seqn, noise_level_map):
        feature_device = torch.device('cuda')
        N, T, C, H, W = seqn.shape

        forward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        backward_hs = torch.empty((N, T, self.num_channels, H, W), device=feature_device)
        seqdn = torch.empty(N, T, C+6, H, W, device='cuda')

        # extract optical flow
        flows_backward, flows_forward = get_flow_2frames(self.pwcnet, seqn)

        # extract forward features
        init_forward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        forward_h = self.forward_rnn(torch.cat((seqn[:, 0], init_forward_h), dim=1))
        forward_hs[:, 0] = forward_h.to(feature_device)
        for i in range(1, T):
            aligned_forward_h, _ = warp(forward_h, flows_forward[:, i-1].cuda())
            forward_h = self.forward_rnn(torch.cat((seqn[:, i], aligned_forward_h), dim=1))
            forward_h = self.concat(torch.cat((forward_h, aligned_forward_h), dim=1))
            forward_hs[:, i] = forward_h.to(feature_device)

        # extract backward features
        init_backward_h = torch.zeros((N, self.num_channels, H, W), device=seqn.device)
        backward_h = self.backward_rnn(torch.cat((seqn[:, -1], init_backward_h), dim=1))
        backward_hs[:, -1] = backward_h.to(feature_device)
        for i in range(2, T+1):
            aligned_backward_h, _ = warp(backward_h, flows_backward[:, T-i].cuda())
            backward_h = self.backward_rnn(torch.cat((seqn[:, T-i],  aligned_backward_h), dim=1))
            backward_h = self.concat(torch.cat((backward_h, aligned_backward_h), dim=1))

            backward_hs[:, T-i] = backward_h.to(feature_device)

        # generate results
        # iterate = if self.training else tqdm(range(T))
        for i in range(T):
            # seqdn[:, i] = self.d(torch.cat((forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device)), dim=1))
            temp = self.d(forward_hs[:, i].to(seqn.device), backward_hs[:, i].to(seqn.device))
            seqdn[:, i] = self.d2(temp)
        flows_backward_p, flows_forward_p = get_flow_2frames(pwcnet_fix, seqdn[:, :, :3, :, :])
        flows_backward, flows_forward = get_flow_2frames_train(self.pwcnet, seqn)

        loss = F.l1_loss(flows_backward_p, flows_backward) + F.l1_loss(flows_forward_p, flows_forward)

        l2_norm = sum(param.norm(2) for param in self.pwcnet.parameters())
        loss = loss + 0.0004 * l2_norm
        # print(loss*0.05)
        return seqdn, None, loss*0.0005
