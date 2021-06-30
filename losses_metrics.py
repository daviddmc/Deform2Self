import torch
import torch.nn as nn
from math import log10
from skimage.metrics import structural_similarity

def masked_mse(x, y, mask=None):
    err2 = (x - y) ** 2
    if mask is not None:
        return torch.sum(mask * err2) / mask.sum()
    else:
        return err2.mean()

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :]) 
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])  

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0

class TotalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight_single = config['weight_single']
        self.weight_registration = config['weight_registration']
        self.weight_grad = config['weight_grad']
        self.weight_multi = config['weight_multi']

    def forward(self, outputs, noisy_img, idx_tgt):
        mask1, denoise1_img, flow, tgt, warp, mask2, denoise2_img = outputs
        if denoise1_img is not None:
            loss_denoise1 = masked_mse(noisy_img, denoise1_img, 1 - mask1)
        else:
            loss_denoise1 = 0
        if warp is not None:
            loss_registration = (masked_mse(warp, tgt) + masked_mse(warp.flip(1), tgt)) / 2
            loss_gradient = gradient_loss(flow)
        else:
            loss_registration = loss_gradient = 0
        if denoise2_img is not None:
            loss_denoise2 = masked_mse(noisy_img[[idx_tgt]], denoise2_img, 1 - mask2)
        else:
            loss_denoise2 = 0
        loss = self.weight_single * loss_denoise1 + \
               self.weight_registration * loss_registration + \
               self.weight_grad * loss_gradient + \
               self.weight_multi * loss_denoise2
        return loss

def psnr(x, y, mask=None):
    if mask is None:
        mse = torch.mean((x - y) ** 2)
    else:
        mse = torch.sum(((x - y) ** 2) * mask) / mask.sum() 
    return 10 * log10(1 / mse.item())

def ssim(x, y, mask=None):
    x = x[0,0].cpu().numpy()
    y = y[0,0].cpu().numpy()
    mssim, S = structural_similarity(x, y, full=True)
    if mask is not None:
        mask = mask[0,0].cpu().numpy()
        return (S * mask).sum() / mask.sum()
    else:
        return mssim