import torch
import torch.nn as nn

def KLDivLoss(output, predict, mean, var):
    mse = nn.MSELoss(reduction='sum')(output, predict)
    kld = -0.5 * torch.sum((1+var-mean.pow(2)-torch.exp(var)).view(-1))
    return mse + kld, mse.item(), kld.item()