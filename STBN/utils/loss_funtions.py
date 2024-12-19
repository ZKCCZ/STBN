import torch
import torch.nn.functional as F
import skimage.restoration as skr


def loss_function(output, truth, mode="loglike", sigma=25):
    if(mode == "mse"):
        loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
    elif(mode == "loglike"):
        eps = 1e-5
        N,C,H,W = truth.shape
        mean = output[0:N, 0:C, 0:H, 0:W].permute(0,2,3,1).reshape(N, H, W, C, 1)
        var = output[0:N, C:C+int(C*(C+1)/2), 0:H, 0:W].permute(0,2,3,1)
        truth = truth.permute(0,2,3,1).reshape(N, H, W, C, 1)
        ax = torch.zeros(N, H, W, int(C*C)).to(output.device)
        I = torch.eye(C).reshape(1, 1, 1, C, C).repeat(N, H, W, 1, 1).to(output.device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, H, W, C, C)
        sigma2I = (((sigma**2)+eps)*I.permute(1,2,3,4,0)).permute(4,0,1,2,3)
        variance = torch.matmul(ax.transpose(3,4), ax) + sigma2I #(sigma**2)*I
        likelihood = 0.5*torch.matmul(torch.matmul((truth-mean).transpose(3,4), torch.inverse(variance)), (truth-mean))
        likelihood = likelihood.reshape(N,H,W)
        likelihood += 0.5*torch.log(torch.det(variance))
        loss = torch.mean(likelihood)
        # loss = torch.mean(likelihood.mean(dim=(1,2)) - 0.1*sigma) 
    return loss


def loss_function_batch(output, truth, mode="loglike", sigma=25):
    if (mode == "mse"):
        loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
    elif (mode == "loglike"):
        eps = 1e-5
        output, truth = output.permute(0, 2, 1, 3, 4), truth.permute(0, 2, 1, 3, 4)
        N, C, T, H, W = truth.shape
        mean = output[0:N, 0:C, 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
        var = output[0:N, C:C+int(C*(C+1)/2), 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1)
        truth = truth.permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
        ax = torch.zeros(N, T, H, W, int(C*C)).to(output.device)
        I = torch.eye(C).reshape(1, 1, 1, 1, C, C).repeat(N, T, H, W, 1, 1).to(output.device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:T, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:T, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, T, H, W, C, C)
        # print(sigma.shape)
        # print(I.shape)
        sigma2I = (((sigma**2)+eps)*I.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3,4)
        variance = torch.matmul(ax.transpose(4, 5), ax) + sigma2I  # (sigma**2)*I

        likelihood = 0.5*torch.matmul(torch.matmul((truth-mean).transpose(4, 5), torch.inverse(variance)), (truth-mean))
        likelihood = likelihood.reshape(N, T, H, W)
        likelihood += 0.5*torch.log(torch.det(variance))
        loss = torch.mean(likelihood)
        # loss = torch.mean(likelihood.mean(dim=(1, 2 ,3)) - 0.1*sigma)
    return loss


def loss_function_batch_z(output, truth, mode="loglike", sigma=25):
    if (mode == "mse"):
        loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
    elif (mode == "loglike"):
        eps = 1e-5
        output, truth = output.permute(0, 2, 1, 3, 4), truth.permute(0, 2, 1, 3, 4)
        N, C, T, H, W = truth.shape
        mean = output[0:N, 0:C, 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
        var = output[0:N, C:C+int(C*(C+1)/2), 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1)
        truth = truth.permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
        ax = torch.zeros(N, T, H, W, int(C*C)).to(output.device)
        I = torch.eye(C).reshape(1, 1, 1, 1, C, C).repeat(N, T, H, W, 1, 1).to(output.device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:T, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:T, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, T, H, W, C, C)
        sigma2I = (((sigma**2)+eps)*I.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)
        variance = torch.matmul(ax.transpose(4, 5), ax) + sigma2I  # (sigma**2)*I

        likelihood = 0.5*torch.matmul(torch.matmul((truth-mean).transpose(4, 5), torch.inverse(variance)), (truth-mean))
        likelihood = likelihood.reshape(N, T, H, W)
        likelihood += 0.5*torch.log(torch.det(variance))
        # loss = torch.mean(likelihood)
        loss = torch.mean(likelihood.mean(dim=(1, 2 ,3)) - 0.1*sigma)
    return loss


def loss_function_batch_mix(output, truth, mode="loglike", std_g=25, std_p=25):
    if (mode == "mse"):
        loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
    elif (mode == "loglike"):
        eps = 1e-5
        output, truth = output.permute(0, 2, 1, 3, 4), truth.permute(0, 2, 1, 3, 4)
        N, C, T, H, W = truth.shape
        mean = output[0:N, 0:C, 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
        var = output[0:N, C:C+int(C*(C+1)/2), 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1)
        truth = truth.permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
        
        ax = torch.zeros(N, T, H, W, int(C*C)).to(output.device)
        I = torch.eye(C).reshape(1, 1, 1, 1, C, C).repeat(N, T, H, W, 1, 1).to(output.device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:T, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:T, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, T, H, W, C, C)
        sigma2I = (((std_g**2)+eps)*I.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)
        variance1 = torch.matmul(ax.transpose(4, 5), ax) + sigma2I  # (sigma**2)*I

        var = output[0:N, C+int(C*(C+1)/2):C+2*int(C*(C+1)/2), 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1)
        ax = torch.zeros(N, T, H, W, int(C*C)).to(output.device)
        I = torch.eye(C).reshape(1, 1, 1, 1, C, C).repeat(N, T, H, W, 1, 1).to(output.device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:T, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:T, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, T, H, W, C, C)
        sigma2I = (((std_p**2)+eps)*I.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)
        variance2 = torch.matmul(ax.transpose(4, 5), ax) + sigma2I  # (sigma**2)*I
        variance = variance1+variance2
        likelihood = 0.5*torch.matmul(torch.matmul((truth-mean).transpose(4, 5), torch.inverse(variance)), (truth-mean))
        likelihood = likelihood.reshape(N, T, H, W)
        likelihood += 0.5*torch.log(torch.det(variance))
        loss = torch.mean(likelihood)
        # loss = torch.mean(likelihood.mean(dim=(1, 2 ,3)) - 0.1*sigma)
    return loss


def loss_function_batch_raw(output, truth, mode="loglike", sigma=25):
    if (mode == "mse"):
        loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
    elif (mode == "loglike"):
        eps = 1e-5
        output, truth = output.permute(0, 2, 1, 3, 4), truth.permute(0, 2, 1, 3, 4)
        sigma = sigma.permute(0, 2, 1, 3, 4)
        N, C, T, H, W = truth.shape
        mean = output[0:N, 0:C, 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
        var = output[0:N, C:C+int(C*(C+1)/2), 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1)
        truth = truth.permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
        ax = torch.zeros(N, T, H, W, int(C*C)).to(output.device)
        I = torch.eye(C).reshape(1, 1, 1, 1, C, C).repeat(N, T, H, W, 1, 1).to(output.device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:T, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:T, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, T, H, W, C, C)
        sigma2I = (((sigma**2)+eps).permute(0, 2, 3, 4, 1)[..., None])*I#.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)
        # sigma2I = ((((sigma**2)+eps).permute(0, 2, 3, 4, 1)[..., None])*I.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)
        variance = torch.matmul(ax.transpose(4, 5), ax) + sigma2I  # (sigma**2)*I

        likelihood = 0.5*torch.matmul(torch.matmul((truth-mean).transpose(4, 5), torch.inverse(variance)), (truth-mean))
        likelihood = likelihood.reshape(N, T, H, W)
        likelihood += 0.5*torch.log(torch.det(variance))
        loss = torch.mean(likelihood)
        # loss = torch.mean(likelihood.mean(dim=(1, 2 ,3)) - 0.1*sigma)
    return loss
