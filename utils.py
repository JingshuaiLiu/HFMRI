import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

from torchvision.utils import save_image

from torchsample.transforms import *

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
    def __call__(self, x):
        shape = x.shape
        outputs = x.clone()
        assert len(shape) == 4, 'inputs must be of shape (B,C,H,W)'
        #
        self.clock = random_interger(low=self.low, high=self.high, shape=(shape[0],)) # a tensor containing rotation factors
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
    def __call__(self, x):
        shape = x.shape
        outputs = x.clone()
        assert len(shape) == 4, 'inputs must be of shape (B,C,H,W)'
        #
        self.translate_h = random_interger(low=-self.h_pixel, high=self.h_pixel, shape=(shape[0],)) # a tensor containing translation pixel units
        self.translate_v = random_interger(low=-self.v_pixel, high=self.v_pixel, shape=(shape[0],)) # a tensor containing translation pixel units
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

''''''

# spectral norm
class Spectral_norm_():
    def __init__(self, n_power_iterations=1):
        self.n_power_iterations = n_power_iterations
    
    def add_spectralNorm(self,m):
        if (m.__class__.__name__.find('Linear') != -1) or (m.__class__.__name__.find('Conv') != -1):
            m = nn.utils.spectral_norm(m, n_power_iterations=self.n_power_iterations) # add to all Conv and Linear classes 
    
    def remove_spectralNorm(self,m):
        if (m.__class__.__name__.find('Linear') != -1) or (m.__class__.__name__.find('Conv') != -1):
            nn.utils.remove_spectral_norm(m)  
            
#net.apply(Spectral_norm_().add_spectralNorm)
#net.apply(Spectral_norm_().remove_spectralNorm)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.orthogonal_(m.weight.data)
        m.weight.data *= 0.1
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Weight_init():
    def __init__(self, scale=1.):
        self.scale = scale

    def init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            m.weight.data *= self.scale # scale initilization
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.orthogonal_(m.weight.data)
            m.weight.data *= self.scale # scale initilization
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

def add_affine(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.bias = None
    elif classname.find("Norm2d") != -1:
        m.affine = True
        m.weight = nn.Parameter(torch.ones(m.num_features))
        m.bias = nn.Parameter(torch.zeros(m.num_features))

class HLoss(nn.Module):
    '''calculate the entropy along the last dimension'''
    def __init__(self, softmax=False):
        super(HLoss, self).__init__()
        self.softmax = softmax

    def forward(self, x):
        if self.softmax:
            b = nn.functional.softmax(x, dim=-1) * nn.functional.log_softmax(x, dim=-1)
            b = -1.0 * b.sum(-1)
        else:
            b = x * torch.log(x)
            b = -1.0 * b.sum(-1)
        return b

def normalize2d(x, eps=1.e-05):
    # mean = x.mean(dim=(-1,-2),keepdim=True).detach()
    # std = x.std(dim=(-1,-2),keepdim=True).detach()

    assert len(x.shape) == 5 and x.shape[2] == 2
    mean = x.mean(dim=(1, -1,-2), keepdim=True).detach() # to resist different shiftings and recalibarate background, i.e. zero points
    std = torch.sqrt(((x-mean)**2).mean((-1, -2), True)) # attention-wise std

    return (x - mean) / torch.sqrt(std**2 + eps)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

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
    x_under = x_under.permute(0, 2, 3, 1)

    assert x_under.size(-1) == 2

    x_under = ifftshift(x_under, dim=(-3, -2))
    x_under = torch.fft(x_under, 2, normalized=True)
    x_under = fftshift(x_under, dim=(-3, -2))

    x_under = x_under.permute(0, 3, 1, 2)
    return x_under

def torch_ifft(x_under):
    x_under = x_under.permute(0, 2, 3, 1)
    assert x_under.size(-1) == 2

    x_under = ifftshift(x_under, dim=(-3, -2))
    x_under = torch.ifft(x_under, 2, normalized=True)
    x_under = fftshift(x_under, dim=(-3, -2))

    x_under = x_under.permute(0, 3, 1, 2)
    return x_under 


def torch_fft3d(x_under):
    # input (B, 2, D, H, W)
    x_under = x_under.permute(0, 2, 3, 4, 1) # (B, D, H, W, 2)
    assert x_under.size(-1) == 2
    #
    x_under = ifftshift(x_under, dim=(-4, -3, -2))
    x_under = torch.fft(x_under, 3, normalized=True)
    x_under = fftshift(x_under, dim=(-4, -3, -2))
    #
    x_under = x_under.permute(0, 4, 1, 2, 3) # back to (B, 2, D, H, W)
    return x_under

def torch_ifft3d(x_under):
    # input (B, 2, D, H, W)
    x_under = x_under.permute(0, 2, 3, 4, 1) # (B, D, H, W, 2)
    assert x_under.size(-1) == 2
    #
    x_under = ifftshift(x_under, dim=(-4, -3, -2))
    x_under = torch.ifft(x_under, 3, normalized=True)
    x_under = fftshift(x_under, dim=(-4, -3, -2))
    #
    x_under = x_under.permute(0, 4, 1, 2, 3) # back to (B, 2, D, H, W)
    return x_under 

def sigtoimage(sig):
    ''' input (B, 2, H, W), and output (B, 1, H, W)  in magnitude '''
    x_real = torch.unsqueeze(sig[:, 0, :, :], 1)
    x_imag = torch.unsqueeze(sig[:, 1, :, :], 1)
    x_image = torch.sqrt(x_real * x_real + x_imag * x_imag)
    return x_image

