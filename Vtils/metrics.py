#!/usr/bin/env python
"""help functions"""

__author__ = "Qiaoying Huang"
__date__ = "04/08/2019"
__institute__ = "Rutgers University"


import torch
import numpy as np
import math
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F

from torchsample.transforms import *


# ifftshift
def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


# calculate RMSE value
def get_rmse(prediction, target):
    diff = torch.abs(target - prediction)
    diff = diff ** 2
    diff = torch.mean(diff)
    diff = torch.sqrt(diff)
    return diff


# calculate PSNR value
def get_psnr(prediction, target):
    mse = torch.mean((prediction - target) ** 2)
    if mse == 0:
        return 0.5
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


### code modified from https://github.com/Po-Hsun-Su/pytorch-ssim ###

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# calculate SSIM value
def get_ssim(prediction, target, window_size=11, in_channel=1):
    ''' input (B, 1, H, W) in range (0, 1) if possible'''
    window_size = window_size
    size_average = True
    channel = in_channel
    window = create_window(window_size, channel)

    (_, channel, _, _) = prediction.size()

    if prediction.is_cuda:
        window = window.cuda(prediction.get_device())
    window = window.type_as(prediction)

    return _ssim(prediction, target, window, window_size, channel, size_average)

############################################################


# Two channels image to magnitude image
def sigtoimage(sig):
    x_real = torch.unsqueeze(sig[:, 0, :, :], 1)
    x_imag = torch.unsqueeze(sig[:, 1, :, :], 1)
    x_image = torch.sqrt(x_real * x_real + x_imag * x_imag)
    return x_image

# fftshift and ifftshift
def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)
    
def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)

def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def torch_fft(x_under):
    x_under_per = x_under.permute(0, 2, 3, 1)

    assert x_under_per.size(-1) == 2

    x_under_per = ifftshift(x_under_per, dim=(-3, -2))
    x_zf_per = torch.fft(x_under_per, 2, normalized=True)
    x_zf_per = fftshift(x_zf_per, dim=(-3, -2)) # position 0 to the center when odd number is given

    x_zf = x_zf_per.permute(0, 3, 1, 2)
    return x_zf

def torch_ifft(x_under):
    x_under_per = x_under.permute(0, 2, 3, 1)
    assert x_under_per.size(-1) == 2

    x_under_per = ifftshift(x_under_per, dim=(-3, -2))
    x_zf_per = torch.ifft(x_under_per, 2, normalized=True)
    x_zf_per = fftshift(x_zf_per, dim=(-3, -2))

    x_zf = x_zf_per.permute(0, 3, 1, 2)
    return x_zf    

# show images
def show_recons(image, kdata, reconsimage):
    diff = (image - reconsimage[-1]).abs() # (B, 1 ,H, W), len(reconsimage)=L
    image_u = sigtoimage(torch_ifft(kdata)) # (B, 1 ,H, W)
    reconsimage = torch.cat(reconsimage, dim=-1) # (B, 1 ,H, L*W)

    recons_line = torch.cat([image_u, reconsimage, image, diff], dim=-1) # (B, 1, H, (L+3)*W)
    recons_line = torch.cat([x for x in recons_line], dim=-2) # (1, B*H, (L+3)*W)
    
    return recons_line.view(1, *recons_line.shape)

'''affine data augmentation'''
# data augmentation
def random_interger(low,high,shape):
    # return torch.randint(low, high, shape)
    return torch.Tensor(*shape).random_(low,high)

class Random_Rotate(object):
    def __init__(self, low=-10, high=10, p=0.5):
        '''low:anti-clock-wise, high:clock-wise'''
        self.low = low 
        self.high = high
        self.p = 0.5
    #    
    def __call__(self, x, clock=None):
        shape = x.shape
        outputs = x.clone()
        assert len(shape) == 4, 'inputs must be of shape (B,C,H,W)'
        #
        self.clock = random_interger(low=self.low, high=self.high, shape=(shape[0],)) if clock is None else clock # a tensor containing rotation factors
        #
        for i in range(shape[0]):
            p_r = random.random()
            if p_r < self.p:
                outputs[i] = Rotate(self.clock[i])(x[i])
        return outputs


class Random_Flip(object):
    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.
        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p
        v : boolean
            whether to vertically flip w/ probability p
        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.p = p 
    def __call__(self, x):
        shape = x.shape
        outputs = x.clone()
        assert len(shape) == 4, 'inputs must be of shape (B,C,H,W)'
        #
        if self.horizontal:
            for i in range(shape[0]):
                p_h = random.random()
                if p_h < self.p:
                    outputs[i] = torch.flip(x[i], dims=[-1])
        #
        outputs_ = outputs.clone()
        if self.vertical:
            for i in range(shape[0]):
                p_v = random.random()
                if p_v < self.p:
                    outputs_[i] = torch.flip(outputs[i], dims=[-2])            
        return outputs_          

class Random_Translate(object):
    def __init__(self, h=True, v=False, h_pixel=5,  v_pixel=5, p=0.5):
        """
        Randomly translate an image horizontally and/or vertically with
        some probability by pixel units
        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p
        v : boolean
            whether to vertically flip w/ probability p
        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.h_pixel = h_pixel
        self.v_pixel = v_pixel
        self.p = p 
    def __call__(self, x, translate_h=None, translate_v=None):
        shape = x.shape
        outputs = x.clone()
        assert len(shape) == 4, 'inputs must be of shape (B,C,H,W)'
        #
        self.translate_h = random_interger(low=-self.h_pixel, high=self.h_pixel, shape=(shape[0],)) if translate_h is None else translate_h # a tensor containing translation pixel units
        
        self.translate_v = random_interger(low=-self.v_pixel, high=self.v_pixel, shape=(shape[0],)) if translate_v is None else translate_v # a tensor containing translation pixel units
        #
        if self.horizontal:
            for i in range(shape[0]):
                p_h = random.random()
                if p_h < self.p:
                    if self.translate_h[i] == 0:
                        pass
                    elif self.translate_h[i] > 0:    
                        outputs[i,:,:,int(self.translate_h[i]):] = x[i,:,:,:-int(self.translate_h[i])]
                        outputs[i,:,:,:int(self.translate_h[i])] = 0.
                    else:
                        outputs[i,:,:,:int(self.translate_h[i])] = x[i,:,:,-int(self.translate_h[i]):]
                        outputs[i,:,:,int(self.translate_h[i]):] = 0.

        #
        outputs_ = outputs.clone()
        if self.vertical:
            for i in range(shape[0]):
                p_v = random.random()
                if p_v < self.p:
                    if self.translate_v[i] == 0:
                        pass
                    elif self.translate_v[i] > 0:    
                        outputs_[i,:,int(self.translate_v[i]):] = outputs[i,:,:-int(self.translate_v[i])]
                        outputs_[i,:,:int(self.translate_v[i])] = 0.
                    else:
                        outputs_[i,:,:int(self.translate_v[i])] = outputs[i,:,-int(self.translate_v[i]):]
                        outputs_[i,:,int(self.translate_v[i]):] = 0.
        return outputs_    