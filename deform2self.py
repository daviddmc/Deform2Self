import torch
import numpy as np
from models import MainModel
from losses_metrics import TotalLoss
from tqdm import tqdm
    
def data_augmentation(img, flip_v, flip_h):
    axis = []
    if flip_v:
        axis.append(2)
    if flip_h:
        axis.append(3)
    if len(axis):
        img = torch.flip(img, axis)
    return img

def deform2self(noisy, idx_tgt, config): # Tx1xHxW

    mode = ''
    if config['enable_single']:
        mode += 'S'
    if config['enable_registration']:
        mode += 'R'
    if config['enable_multi']:
        mode += 'M'

    if mode == 'S':
        noisy_img = noisy[[idx_tgt]]
        idx_tgt = 0

    model = MainModel(config['prob_mask'], noisy.shape[0], mode)
    if config['gpu']:
        noisy = noisy.cuda()
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    total_loss = TotalLoss(config)

    if config['verbose']:
        print('Deform2Self training')
    for _ in tqdm(range(config['num_train']), disable=not config['verbose']):
        flip_v, flip_h = np.random.choice(2, size=2)
        noisy_img = data_augmentation(noisy, flip_v, flip_h) # T x 1 x H x W
        loss = total_loss(model(noisy_img, idx_tgt), noisy_img, idx_tgt)
        optimizer.zero_grad()
        loss.backward()
        if config['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
        optimizer.step()

    if config['verbose']:
        print('Deform2Self inference')
    with torch.no_grad():
        output_img_all = 0
        for _ in tqdm(range(config['num_test']), disable=not config['verbose']) :
            flip_v, flip_h = np.random.choice(2, size=2)
            noisy_img = data_augmentation(noisy, flip_v, flip_h)
            _, denoise1_img, _, _, _, _, denoise2_img = model(noisy_img, idx_tgt)
            output_img = data_augmentation(denoise2_img if denoise2_img is not None else denoise1_img, flip_v, flip_h)
            output_img_all += output_img
        output_img = output_img_all / config['num_test']
    
    return output_img
   

def deform2self_sequence(noisy, config):
    denoised = []
    k = config['num_slice']
    for i in range(noisy.shape[0]):
        if config['verbose']:
            print('processing slice (%d/%d)' % (i+1, noisy.shape[0]))
        i_min = max(0, i-k)
        i_max = min(noisy.shape[0], i+k+1)
        denoised.append(deform2self(noisy[i_min:i_max], i-i_min, config))
    return torch.cat(denoised, 0)
        