import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, dim, c_in, c_out, stride=1, p_drop=0, alpha=0.1):
        super().__init__()
        self.p_drop = p_drop
        self.alpha = alpha
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')
        self.conv = conv_fn(c_in, c_out, ksize, stride, 1)

    def forward(self, x, mask=None):
        if self.p_drop:
            x = F.dropout(x, self.p_drop)
        if mask is not None:
            x = self.conv(x, mask)
        else:
            x = self.conv(x)
        x = F.leaky_relu(x, self.alpha)
        return x


class RegistrationNet(nn.Module):
    def __init__(self, 
                 dim=2,
                 c = 2, 
                 enc_nf=(16, 32, 32, 32), 
                 dec_nf=(32, 32, 32, 32, 32, 16, 16),
                 full_size=True):
        super(RegistrationNet, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        c_in = 2 * c

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = c_in if i == 0 else enc_nf[i-1]
            self.enc.append(ConvBlock(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(ConvBlock(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(ConvBlock(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(ConvBlock(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(ConvBlock(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(ConvBlock(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(ConvBlock(dim, dec_nf[4] + c_in, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = ConvBlock(dim, dec_nf[5], dec_nf[6]) 
 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
             y = self.dec[i](y)
             y = self.upsample(y)
             y = torch.cat([y, x_enc[-(i+2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
             y = self.upsample(y)
             y = torch.cat([y, x_enc[0]], dim=1)
             y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
             y = self.vm2_conv(y)
             
        y = self.flow(y)

        return y
        
        
class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        self.mode = mode
        
    def gen_grid(self, flow):
        vectors = [torch.arange(0, s, device=flow.device, dtype=flow.dtype) for s in flow.shape[2:]] 
        grids = torch.meshgrid(vectors) 
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        #grid = grid.type(torch.FloatTensor)
        return grid

    def forward(self, flow, *args):   
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.gen_grid(flow) + flow 

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) 
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
            new_locs = new_locs[..., [2,1,0]]
            
        if len(args) == 1:
            return F.grid_sample(args[0], new_locs, mode=self.mode, align_corners=True)
        else:
            return tuple(F.grid_sample(arg, new_locs, mode=self.mode, align_corners=True) for arg in args)


class DenoiseNet(nn.Module):

    def __init__(self, c_in, c_feat=48, c_out=1, p_drop=0.3):
        super().__init__()
        self.n_down = 5
        convs = [ConvBlock(2, c_in, c_feat, p_drop=0)]
        for i in range(self.n_down+1):
            convs.append(ConvBlock(2, c_feat, c_feat, p_drop=0))
        for i in range(self.n_down-1):
            convs.append(ConvBlock(2, c_feat*(3 if i else 2), c_feat*2, p_drop=p_drop))
            convs.append(ConvBlock(2, c_feat*2, c_feat*2, p_drop=p_drop))
        convs.append(ConvBlock(2, 2*c_feat+c_in, 64, p_drop=p_drop))
        convs.append(ConvBlock(2, 64, 32, p_drop=p_drop))
        self.convs = nn.ModuleList(convs)
        self.conv_out = nn.Conv2d(32, c_out, kernel_size=3, padding=1)

    def forward(self, x, mask=None):

        skips = [x]

        if mask is not None:
            x = self.convs[0](x, mask)
        else:
            x = self.convs[0](x)
        for i in range(self.n_down):
            x = self.convs[i+1](x)
            x = F.max_pool2d(x, kernel_size=2)
            skips.append(x)

        x = self.convs[self.n_down+1](skips.pop())

        for i in range(self.n_down):
            x = F.interpolate(x, scale_factor=2)
            x = torch.cat([x, skips.pop()], 1)
            x = self.convs[self.n_down+2+2*i](x)
            x = self.convs[self.n_down+3+2*i](x)

        x = self.conv_out(x)
        #x = torch.sigmoid(x)

        return x

class MainModel(nn.Module):

    def __init__(self, mask_p, N_slice, mode='SRM'):
        super().__init__()
        
        assert mode in ['S', 'SM', 'RM', 'M', 'SRM']
        self.mode = mode
        self.mask_p = mask_p
        
        if 'S' in self.mode:
            self.denoise1 = DenoiseNet(1)
        if 'R' in self.mode:
            self.registration = RegistrationNet(c = 2 if 'S' in self.mode else 1)
            self.transform = SpatialTransformer()
        if 'M' in self.mode:
            self.denoise2 = DenoiseNet(N_slice * (2 if 'S' in self.mode else 1))
            
    def forward(self, noisy_img, idx_tgt): # T x 1 x H x W
        T, _, H, W = noisy_img.shape
        
        mask1 = denoise1_img = flow = tgt = warp = mask2 = denoise2_img = None
        
        for stage in self.mode:
            if stage == 'S':
                # denoise (single)
                mask1 = (nn.init.uniform_(torch.zeros(T, 1, H, W, device=noisy_img.device)) > self.mask_p).float() # T x N x H x W
                input_img = noisy_img * mask1 # T x N x H x W
                input_img = torch.reshape(input_img, (-1, 1, H, W)) # TN x 1 x H x W
                denoise1_img = self.denoise1(input_img) # TN x 1 x H x W
                denoise1_img = torch.reshape(denoise1_img, (T, -1, H, W)) # T x N x H x W
            elif stage == 'R':
                # registration
                if denoise1_img is not None:
                    input_img = torch.cat((denoise1_img.mean(dim=1, keepdim=True).detach_(), noisy_img), 1) # T x 2 x H x W  (denoise11, noisy)
                else:
                    input_img = noisy_img
                idx = list(range(idx_tgt)) + list(range(idx_tgt+1, T))
                input_img = torch.cat((input_img[idx], input_img[[idx_tgt]*len(idx)]), 1) # T-1 x 4 x H x W  (denoise11_s, noisy_s, denoise11_t, noisy_t)
                src = input_img[:, :input_img.shape[1]//2] # T-1 x 2 x H x W
                tgt = input_img[:, input_img.shape[1]//2:] # T-1 x 2 x H x W
                flow = self.registration(input_img)
                warp = self.transform(flow, src) # T-1 x 2 x H x W
            elif stage == 'M':
                # denoise (multiple)
                mask2 = (nn.init.uniform_(torch.zeros(1, 1, H, W, device=noisy_img.device)) > self.mask_p).float() # 1 x 1 x H x W
                if warp is not None: # w/ R
                    input_img = torch.cat((tgt[:1], warp)).detach_() # T x 2 x H x W  (1 x 2 x H x W + T-1 x 2 x H x W)
                    input_img[:1, -1:] = input_img[:1, -1:] * mask2
                    input_img = torch.flatten(input_img, end_dim=1).unsqueeze(0) # 1 x T2 x H x W
                elif denoise1_img is None: # w/o R, w/o S
                    input_img = noisy_img.permute(1, 0, 2, 3).clone() # 1 x T x H x W
                    input_img[:, idx_tgt] = input_img[:, idx_tgt] * mask2
                else: # w/o R, w/ S
                    input_img = torch.cat((denoise1_img.mean(dim=1, keepdim=True).detach_(), noisy_img), 1) # T x 2 x H x W  (denoise11, noisy)
                    input_img[idx_tgt, -1:] = input_img[idx_tgt, -1:] * mask2
                    input_img = torch.flatten(input_img, end_dim=1).unsqueeze(0) # 1 x T2 x H x W
                denoise2_img = self.denoise2(input_img) # 1 x 1 x H x W
            else:
                raise Exception('unknown stage')
        
        return  mask1, denoise1_img, flow, tgt, warp, mask2, denoise2_img