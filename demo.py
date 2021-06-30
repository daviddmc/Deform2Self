from PIL import Image
import os
import numpy as np
import torch
from deform2self import deform2self, deform2self_sequence
from utils import add_gaussian_noise, img_float2int
import yaml


if __name__ == '__main__':

    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu']

    imgs = []
    for fname in sorted(os.listdir('./data')):
        imgs.append(np.array(Image.open(os.path.join('./data', fname))))
    imgs = np.stack(imgs, axis=0).astype(np.float32)
    imgs /= imgs.max()
    imgs = torch.from_numpy(imgs).unsqueeze(1) # Tx1xHxW

    noisy = add_gaussian_noise(imgs, 15)
    
    denoised = deform2self_sequence(noisy, config) # Tx1xHxW

    denoised = denoised.squeeze(1).cpu().numpy()
    denoised = img_float2int(denoised)

    noisy = noisy.squeeze(1).numpy()
    noisy = img_float2int(noisy)

    for i, (img_n, img_d) in enumerate(zip(noisy, denoised)):
        Image.fromarray(img_n).save('outputs/noisy%d.png' % i)
        Image.fromarray(img_d).save('outputs/denoised%d.png' % i)
    






     