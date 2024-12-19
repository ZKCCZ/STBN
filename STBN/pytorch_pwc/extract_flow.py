import math
import torch

# im1_torch, im2_torch in shape (N, C, H, W)
def extract_flow_torch(model, im1_torch, im2_torch):
    # interpolate image, make new_H, mew_W divide by 64
    assert im1_torch.shape == im2_torch.shape
    N, C, H, W = im1_torch.shape
    device = im1_torch.device
    new_H = int(math.floor(math.ceil(H / 64.0) * 64.0))
    new_W = int(math.floor(math.ceil(W / 64.0) * 64.0))
    im1_torch = torch.nn.functional.interpolate(input=im1_torch, size=(new_H, new_W), mode='bilinear',
                                                 align_corners=False)
    im2_torch = torch.nn.functional.interpolate(input=im2_torch, size=(new_H, new_W), mode='bilinear',
                                                 align_corners=False)
    model.eval()
    with torch.no_grad():
        flo12 = model(im1_torch, im2_torch)
    flo12 = 20.0 * torch.nn.functional.interpolate(input=flo12, size=(H, W), mode='bilinear',
                                                          align_corners=False)
    flo12[:, 0, :, :] *= float(W) / float(new_W)
    flo12[:, 1, :, :] *= float(H) / float(new_H)
    return flo12

# im1_np, im2_np in shape (C, H, W)
def extract_flow_np(model, im1_np, im2_np):
    im1_torch = torch.from_numpy(im1_np).unsqueeze(0).to(torch.device('cuda'))
    im2_torch = torch.from_numpy(im2_np).unsqueeze(0).to(torch.device('cuda'))
    flo12_torch = extract_flow_torch(model, im1_torch, im2_torch)
    flo12_np = flo12_torch.detach().cpu().squeeze(0).numpy()
    return flo12_np


def get_flow_2frames(model, x):
    b, n, c, h, w = x.size()
    new_H = int(math.floor(math.ceil(h / 64.0) * 64.0))
    new_W = int(math.floor(math.ceil(w / 64.0) * 64.0))
    x = torch.nn.functional.interpolate(input=x.view(-1,c,h,w), size=(new_H, new_W), mode='bilinear',
                                        align_corners=False).view(b, n, c, new_H, new_W)
    model.eval()
    with torch.no_grad():
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, new_H, new_W)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, new_H, new_W)
        # backward
        flows_backward = model(x_1, x_2)
        # print(flows_backward.shape)
        # forward
        flows_forward = model(x_2, x_1)

        flows_backward = 20.0 * torch.nn.functional.interpolate(input=flows_backward, size=(h, w), mode='bilinear',
                                                       align_corners=False)
        flows_backward[:, 0, :, :] *= float(w) / float(new_W)
        flows_backward[:, 1, :, :] *= float(h) / float(new_H)
        
        flows_forward = 20.0 * torch.nn.functional.interpolate(input=flows_forward, size=(h, w), mode='bilinear',
                                                                align_corners=False)
        flows_forward[:, 0, :, :] *= float(w) / float(new_W)
        flows_forward[:, 1, :, :] *= float(h) / float(new_H)


        return flows_backward.view(b, n-1, 2, h, w), flows_forward.view(b, n-1, 2, h, w)


def get_flow_2frames_train(model, x):
    b, n, c, h, w = x.size()
    new_H = int(math.floor(math.ceil(h / 64.0) * 64.0))
    new_W = int(math.floor(math.ceil(w / 64.0) * 64.0))
    x = torch.nn.functional.interpolate(input=x.view(-1, c, h, w), size=(new_H, new_W), mode='bilinear',
                                        align_corners=False).view(b, n, c, new_H, new_W)

    x_1 = x[:, :-1, :, :, :].reshape(-1, c, new_H, new_W)
    x_2 = x[:, 1:, :, :, :].reshape(-1, c, new_H, new_W)
    # backward
    flows_backward = model(x_1, x_2)
    # print(flows_backward.shape)
    # forward
    flows_forward = model(x_2, x_1)

    flows_backward = 20.0 * torch.nn.functional.interpolate(input=flows_backward, size=(h, w), mode='bilinear',
                                                            align_corners=False)
    flows_backward[:, 0, :, :] *= float(w) / float(new_W)
    flows_backward[:, 1, :, :] *= float(h) / float(new_H)

    flows_forward = 20.0 * torch.nn.functional.interpolate(input=flows_forward, size=(h, w), mode='bilinear',
                                                            align_corners=False)
    flows_forward[:, 0, :, :] *= float(w) / float(new_W)
    flows_forward[:, 1, :, :] *= float(h) / float(new_H)

    return flows_backward.view(b, n-1, 2, h, w), flows_forward.view(b, n-1, 2, h, w)
