import torch
import numpy as np

noise_g = np.random.RandomState(233)

def add_gaussian_noise(img, sigma): # img [0,1]
    #noise = torch.zeros_like(img)
    if sigma > 0:
        noise = noise_g.randn(*img.shape) * (sigma / 255.)
        noise = torch.tensor(noise, dtype=img.dtype, device=img.device)
    noisy_img = img + noise
    return noisy_img

def add_rician_noise(img, sigma):
    noisy_img = img
    if sigma > 0:
        noise1 = noise_g.randn(*img.shape) * (sigma / 255.)
        noise1 = torch.tensor(noise1, dtype=img.dtype, device=img.device)
        noise2 = noise_g.randn(*img.shape) * (sigma / 255.)
        noise2 = torch.tensor(noise2, dtype=img.dtype, device=img.device)
        noise = torch.stack((noise1, noise2), -1)
        noisy_img = torch.stack((noisy_img, torch.zeros_like(noisy_img)), -1)
        noisy_img = torch.ifft(torch.fft(noisy_img, 2, True) + noise, 2, True)
        noisy_img = torch.sqrt(noisy_img[...,0]**2 + noisy_img[...,1]**2)
    return noisy_img
    
def add_poisson_noise(img, sigma):
    noisy_img = noise_g.poisson(img.cpu().numpy() * sigma) / sigma
    noisy_img = torch.tensor(noisy_img, dtype=img.dtype, device=img.device)
    return noisy_img
    
def img_float2int(img):
    img = np.clip(img, 0, 1)
    img = (img * 65535).astype(np.uint16)
    return img