import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import skimage.restoration as skr


def post_process(output, input, sigma=25, device="cuda"):

    eps = 1e-5
    N, C, H, W = input.shape
    mean = output[0:N, 0:C, 0:H, 0:W].permute(0, 2, 3, 1).reshape(N, H, W, C, 1)
    var = output[0:N, C:C+int(C*(C+1)/2), 0:H, 0:W].permute(0, 2, 3, 1)
    input = input.permute(0, 2, 3, 1).reshape(N, H, W, C, 1)
    ax = torch.zeros(N, H, W, int(C*C)).to(device)
    I = torch.eye(C).reshape(1, 1, 1, C, C).repeat(N, H, W, 1, 1).to(device)
    idx1 = 0
    for i in range(C):
        idx2 = idx1 + C-i
        ax[0:N, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:H, 0:W, idx1:idx2]
        idx1 = idx2
    ax = ax.reshape(N, H, W, C, C)
    variance = torch.inverse(torch.matmul(ax.transpose(3, 4), ax) + eps*I)
    Ibysigma2 = ((1/((sigma**2)+eps))*I.permute(1, 2, 3, 4, 0)).permute(4, 0, 1, 2, 3)
    inputbysigma2 = ((1/((sigma**2)+eps))*input.permute(1, 2, 3, 4, 0)).permute(4, 0, 1, 2, 3)
#         image = torch.matmul(torch.inverse(variance + (1/(sigma**2))*I),
#                              (torch.matmul(variance, mean) + (1/(sigma**2))*input)).reshape(N, H, W, C).permute(0,3,1,2)
    image = torch.matmul(torch.inverse(variance + Ibysigma2),
                            (torch.matmul(variance, mean) + inputbysigma2)).reshape(N, H, W, C).permute(0, 3, 1, 2)
    input = input.reshape(N, H, W, C).permute(0, 3, 1, 2)
    mean_image = output[0:N, 0:C, 0:H, 0:W]

    return image, mean_image



def post_process_batch(output, input, sigma=25):
    eps = 1e-5
    output, input = output.permute(0, 2, 1, 3, 4), input.permute(0, 2, 1, 3, 4)
    N, C, T, H, W = input.shape
    mean = output[0:N, 0:C, 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
    var = output[0:N, C:C+int(C*(C+1)/2), 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1)
    input = input.permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
    ax = torch.zeros(N, T, H, W, int(C*C)).to(output.device)
    I = torch.eye(C).reshape(1, 1, 1, 1, C, C).repeat(N, T, H, W, 1, 1).to(output.device)
    idx1 = 0
    for i in range(C):
        idx2 = idx1 + C-i
        ax[0:N, 0:T, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:T, 0:H, 0:W, idx1:idx2]
        idx1 = idx2
    ax = ax.reshape(N, T, H, W, C, C)
    variance = torch.inverse(torch.matmul(ax.transpose(4, 5), ax) + eps*I)
    Ibysigma2 = ((1/((sigma**2)+eps))*I.permute(1, 2, 3, 4,5, 0)).permute(5, 0, 1, 2, 3,4)
    inputbysigma2 = ((1/((sigma**2)+eps))*input.permute(1, 2, 3, 4,5, 0)).permute(5, 0, 1, 2, 3,4)

    image = torch.matmul(torch.inverse(variance + Ibysigma2),
                         (torch.matmul(variance, mean) + inputbysigma2)).reshape(N, T, H, W, C).permute(0, 4, 1, 2,3)
    # input = input.reshape(N, T, H, W, C).permute(0, 4, 1, 2, 3)
    mean_image = output[0:N, 0:C,  0:T, 0:H, 0:W]

    return image.permute(0, 2, 1, 3, 4), mean_image.permute(0, 2, 1, 3, 4)


def post_process_batch_raw(output, input, sigma=25):
    eps = 1e-5
    output, input = output.permute(0, 2, 1, 3, 4), input.permute(0, 2, 1, 3, 4)
    sigma = sigma.permute(0, 2, 1, 3, 4)

    N, C, T, H, W = input.shape
    mean = output[0:N, 0:C, 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
    var = output[0:N, C:C+int(C*(C+1)/2), 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1)
    input = input.permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
    ax = torch.zeros(N, T, H, W, int(C*C)).to(output.device)
    I = torch.eye(C).reshape(1, 1, 1, 1, C, C).repeat(N, T, H, W, 1, 1).to(output.device)
    idx1 = 0
    for i in range(C):
        idx2 = idx1 + C-i
        ax[0:N, 0:T, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:T, 0:H, 0:W, idx1:idx2]
        idx1 = idx2
    ax = ax.reshape(N, T, H, W, C, C)
    variance = torch.inverse(torch.matmul(ax.transpose(4, 5), ax) + eps*I)
    # Ibysigma2 = ((1/(((sigma**2).permute(0, 2, 3,4, 1)[..., None])+eps))*I.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)
    Ibysigma2 = (1/(((sigma**2).permute(0, 2, 3,4, 1)[..., None])+eps))*I#.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)

    # inputbysigma2 = ((1/((sigma**2)+eps))*input.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)
    inputbysigma2 = (1/((sigma**2).permute(0, 2, 3, 4, 1)[..., None]+eps))*input  # .permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)

    image = torch.matmul(torch.inverse(variance + Ibysigma2),
                         (torch.matmul(variance, mean) + inputbysigma2)).reshape(N, T, H, W, C).permute(0, 4, 1, 2, 3)
    # input = input.reshape(N, T, H, W, C).permute(0, 4, 1, 2, 3)
    mean_image = output[0:N, 0:C,  0:T, 0:H, 0:W]

    return image.permute(0, 2, 1, 3, 4), mean_image.permute(0, 2, 1, 3, 4)
