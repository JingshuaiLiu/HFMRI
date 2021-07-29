# supervised training
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from adaptive_conv_models import *
from discriminator import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from h5topng.data import transforms as T
from h5topng.common import subsample

from Vtils.pytorch_msssim_master.pytorch_msssim import MS_SSIM, gaussian_filter

from adaptive_conv_models.vtils import Random_Rotate, Random_Flip, Random_Translate

class To_h_space:
    def __init__(self, mask=None, center_fractions=[0.04], accelerations=[8], seed=None):
        self.mask = mask
        self.seed = seed
        if mask == None:
            self.mask_func = subsample.MaskFunc(center_fractions, accelerations)

    def __call__(self, data):
        device = data.device
        # to complex data (B,1,H,W,2)
        data = data.unsqueeze(dim=-1).transpose(1,-1)
        # to fft domian
        data = T.fft2(data)

        # apply mask
        if self.mask == None:
            data, _ = T.apply_mask(data, self.mask_func, seed=self.seed)
        else:
            self.mask = self.mask.to(device)
            data = torch.where(self.mask == 0, torch.Tensor([0.]).to(device), data)
        
        # to image domain
        data = T.ifft2(data)
        return data.transpose(1,-1).squeeze(dim=-1)

class To_k_space:
    def __init__(self, mask=None, center_fractions=[0.04], accelerations=[8], seed=None):
        self.mask = mask
        self.seed = seed
        if mask == None:
            self.mask_func = subsample.MaskFunc(center_fractions, accelerations)

    def __call__(self, data):
        device = data.device
        # to complex data (B,1,H,W,2)
        data = data.unsqueeze(dim=-1).transpose(1,-1)
        # to fft domian
        data = T.fft2(data)

        # apply mask
        if self.mask == None:
            data, _ = T.apply_mask(data, self.mask_func, seed=self.seed)
        else:
            self.mask = self.mask.to(device)
            data = torch.where(self.mask == 0, torch.Tensor([0.]).to(device), data)
        
        # to (B,2,H,W)
        return data.transpose(1,-1).squeeze(dim=-1)

from utils import torch_fft, torch_ifft, sigtoimage, HLoss, normalize2d
class Soft_Data_Consistency(nn.Module):
    '''mask: (B=1, C=1, H, W)'''
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self.mask_c = torch.ones_like(mask) - mask # complementary of support
        
    # def __call__(self, data, data_u):
    def forward(self, data, data_u):
        '''input: (B,2,H,W)'''
        device = data.device
        self.mask = self.mask.to(device)
        self.mask_c = self.mask_c.to(device)

        # # to complex data (B,1,H,W,2)
        # data = data.unsqueeze(dim=-1).transpose(1,-1)
        # data_u = data_u.unsqueeze(dim=-1).transpose(1,-1)

        # # to fft domian
        # data = T.fft2(data)
        # data_u = T.fft2(data_u)

        data = torch_fft(data)
        data_u = torch_fft(data_u)

        # DC operation
        data_dc = data*self.mask_c + data_u*self.mask

        # to image domain
        data_dc = torch_ifft(data_dc)
        # return data_dc.transpose(1,-1).squeeze(dim=-1)
        return data_dc

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="NYU_MRI", help="name of the dataset")
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--mask', default=None, help='path to dataset')
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_depth", type=int, default=1, help="size of image depth, e.g. coils")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--repeat_dim", type=int, default=1, help="number of random samples in test")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--lambda_adv", type=float, default=1., help="pixelwise loss weight")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise reconstruction loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_vgg", type=float, default=1., help="perceptual loss weight")
parser.add_argument("--lambda_grad", type=float, default=10., help="gradient penalty")

parser.add_argument("--mphn", default=False, action='store_true', help="mphn model")
parser.add_argument("--not_ML_dense", default=False, action='store_true', help="multi-level dense architecture")
parser.add_argument("--not_plus", default=False, action='store_true', help="no feature repeation to balance the model parameter size")
parser.add_argument("--dense", default=False, action='store_true', help="dense connections")
parser.add_argument("--stasm", default=False, action='store_true', help="add STASM modules")
parser.add_argument("--stasm_groups", type=int, default=1)

parser.add_argument("--data_consistency", default=False, action='store_true', help="interleaved data consistency")

opt = parser.parse_args()
# print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_depth, opt.img_height, opt.img_width)

# mean square normalize
def mean_square_normalize(data, thresh=0.05, ratio=0.1, dilate=1.0):
    data[data.abs()<thresh] = 0.0 # threshold
    
    shape = data.shape
    mean_square = (data**2).sum(1).sqrt().mean((-2,-1))
    mean_square = mean_square.view((shape[0],1,1,1)).repeat((1,shape[1],shape[2],shape[3]))
    
    # normalize
    data = data/mean_square*ratio 
    data = torch.tanh(data*dilate)
    return data

def sample_images(epoch, i):
    """Saves a generated sample rom the validation set"""
    generator.eval()
    # imgs = next(iter(val_dataloader))
    img_samples = None
    attention_samples = []

    for img_A, img_B in zip(to_cyc(val_dataset.type(Tensor)), val_dataset.type(Tensor)):
    # for img_A, img_B in zip(To_h_space(mask=None)(val_dataset.type(Tensor)), val_dataset.type(Tensor)):
        img_A = img_A.unsqueeze(dim=0) # (1, C, H W)
        img_B = img_B.unsqueeze(dim=0)
        # Repeat input image by number of desired columns
        repeat_dim = opt.repeat_dim
        real_A = img_A.repeat(repeat_dim, 1, 1, 1)
        real_A = Variable(real_A)

        # Generate samples
        with torch.no_grad():
            fake_B, _ = generator(real_A.contiguous().unsqueeze(dim=2), zero_filled=real_A.clone(), csm=None, dc_operator=multi_coil_dc)
            fake_B = fake_B.contiguous().squeeze(dim=2)
        
        '''compute magnitude maps'''
        # (B,2,H,W) to (B,2,H,W,1), B=1
        img_A = img_A.unsqueeze(-1)
        img_B = img_B.unsqueeze(-1)
        fake_B = fake_B.unsqueeze(-1)
        # to complex format (B,1,H,W,2)
        img_A = img_A.transpose(1,-1)
        img_B = img_B.transpose(1,-1)
        fake_B = fake_B.transpose(1,-1)
        # to magnitude in (B,1,H,W)
        img_A = T.complex_abs(img_A)
        img_B = T.complex_abs(img_B)
        fake_B = T.complex_abs(fake_B)

        # diff
        diff = (fake_B-img_B).abs()

        # Concatenate samples horisontally
        fake_B = torch.cat([x for x in fake_B], -1) # (C, H, 2*N*W)
        diff = torch.cat([x for x in diff], -1) # (C, H, 2*N*W)

        img_sample = torch.cat((img_A.squeeze(dim=0), fake_B, img_B.squeeze(dim=0), diff), -1) # (C, H, (N+2)*W)
        img_sample = img_sample.view(1, *img_sample.shape) # (1, C, H, (N+2)*W)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat([img_samples, img_sample], -2) # (1, C, M*H, (N+2)*W)
    # print(img_samples.shape, img_sample.shape)    
    save_image(img_samples, "images/%s/Adap_GAN_epoch_%d_%d.png" % (opt.dataset_name, epoch, i), nrow=8, normalize=False)
    generator.train()


# measurement method to produce real_A from real_B: (1 ,1, 1, 256, 1)
if opt.mask == None:
    mask = opt.mask
else: 
    mask = torch.load(opt.mask)

to_cyc = To_h_space(mask=mask)
to_k = To_k_space(mask=mask)
# to_cyc = To_h_space(mask=None, center_fractions=[0.04], accelerations=[8]) # sampling pattern diversity
# to_k = To_k_space(mask=None, center_fractions=[0.04], accelerations=[8])

soft_dc = Soft_Data_Consistency(mask=mask.squeeze(dim=-1)) # DC opeerator

def multi_coil_dc(inputs, zero_filled, CSM=None):
    outputs = soft_dc(inputs, zero_filled) # data consistency
    return outputs


# Loss functions
# mae_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()

eps = 1e-12
Smooth_L1 = lambda output, target: torch.sqrt((output - target)**2+eps).mean()


ms_ssim = MS_SSIM(data_range=1, channel=2, K=(0.01, 0.03)) # Try a larger K2 constant (e.g. 0.4)
win = ms_ssim.win

# Initialize generator, encoder and discriminators
# generator = AdapGenerator(input_shape)
# D_VAE = RA_MultiDiscriminator([input_shape[0], *input_shape[2:]]) # as we often distinguish among single-coil views

if opt.not_ML_dense:
    generator = Sequential_Dense_Network(img_shape=(2,256,256), out_channel=2, scaler_c=2, dense_dilation=False, stages=3, dense=opt.dense, no_plus = opt.not_plus)
else:
    generator = Multi_Level_Dense_Network(img_shape=(2,256,256), out_channel=2, scaler_c=2, dense_dilation=False, stages=3, stasm=opt.stasm, groups=opt.stasm_groups, data_consistency=opt.data_consistency)

D_VAE =  RA_MultiDiscriminator_CBAM([input_shape[0], *input_shape[2:]], p=0.1)

# D_VAE =  RA_MultiDiscriminator_Unet([input_shape[0], *input_shape[2:]])

# generator = Deep_Projection_Network(input_shape, mask=mask.squeeze(dim=-1))

vgg = models.vgg11_bn(pretrained=True).features[:19].cuda()
for param in vgg.parameters(): 
    param.requires_grad = False # no longer parameter(), but can receive and transmit gradients; it saves computational costs and memory

VGGList = nn.ModuleList()
VGGList.add_module('vgg_0', vgg[:9])
VGGList.add_module('vgg_1', vgg[9:12])
VGGList.add_module('vgg_2', vgg[12:16])
VGGList.add_module('vgg_3', vgg[16:])

from utils import Weight_init

if cuda:
    generator = generator.cuda()
    D_VAE = D_VAE.cuda()
    mae_loss.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'), strict=False)
    D_VAE.load_state_dict(torch.load("saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
    optimizer_G.load_state_dict(torch.load("saved_models/%s/optimizer_G_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
    optimizer_D_VAE.load_state_dict(torch.load("saved_models/%s/optimizer_D_VAE_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# prepare dataset
dataset = torch.load(opt.dataroot) # complex MRI data (B,2,H,W)
start_ = 100
val_dataset = dataset[[10, 30, 35, 55, 75],:,start_:start_+256] # cropped validation samples, range(15,26,5)
# val_dataset = dataset[list(range(10,81,5)),:,start_:start_+256]
dataset = dataset[164:,:,list(range(start_, start_+256))] # cropped training samples

# create dataloaders for training and validation
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

# ----------
#  Training
# ----------
if __name__ == '__main__':
    # Adversarial loss
    valid = 1.
    fake = 0.

    prev_time = time.time()

    for epoch in range(opt.epoch+1, opt.n_epochs+opt.epoch+1):

        for i, batch in enumerate(dataloader):
            '''data augmentation'''
            # Runs the forward pass with autocasting.
            optimizer_G.zero_grad()

            # Runs the forward pass with autocasting.
            with torch.cuda.amp.autocast(enabled=False):
                # Set model input
                real_B = Variable(batch.type(Tensor))
                real_A = Variable(to_cyc(batch.type(Tensor)).detach())

                real_K = Variable(to_k(batch.type(Tensor)).detach())

                # Produce output using real_A
                fake_B, _ = generator(real_A, zero_filled=real_A.clone(), csm=None, dc_operator=multi_coil_dc)

                '''non-uniform mean'''

                # Pixelwise loss of translated image by VAE
                alpha = 0.64 # 0.84

                # L1_loss = torch.sqrt(nn.MSELoss()(fake_B, real_B))
                # L1_loss = (fake_B - real_B).abs()
                L1_loss = torch.sqrt((fake_B - real_B)**2 + eps)
                L1_loss = gaussian_filter(L1_loss, win.to(L1_loss.device)).mean() # Gaussian coefficients indicating the contribution

                # SSIM
                MS_SSIM_Loss = 1. - ms_ssim((fake_B+1.)/2, (real_B+1.)/2)

                # total pixel loss
                loss_pixel = (1-alpha)*L1_loss + alpha*MS_SSIM_Loss

                # Adversarial loss
                loss_VAE_GAN = D_VAE.compute_loss(real_B, fake_B, valid=fake, fake=valid, sg=False) # relativistic average
                # loss_VAE_GAN = D_VAE.compute_loss(fake_B, None, fake=valid, sg=False)

                # feature attention using a U-net-like D
                loss_FA = torch.Tensor(1).fill_(0.).type(Tensor)
                # loss_FA = torch.sqrt(((1.-relative_score.detach())*(fake_B - real_B))**2 + eps).mean()

                # Total Loss (Generator + Encoder)
                loss_GE = opt.lambda_adv*loss_VAE_GAN + opt.lambda_pixel * (loss_pixel + 0.5*loss_FA)

                # ---------
                # cLR-GAN
                # ---------

                loss_latent = opt.lambda_latent * Smooth_L1(to_k(fake_B), real_K)
                # loss_latent = loss_latent.detach()

                # VGG loss
                content_loss = []
                gram_loss = []
                lambda_gram = 0.005

                weight_list = [1., 1.5, 3., 4.5]

                # VGG loss via vgg11_bn
                real_content = sigtoimage(real_B).repeat(1,3,1,1)
                fake_content = sigtoimage(fake_B).repeat(1,3,1,1)

                for k, m in enumerate(VGGList):
                    real_content = m(real_content).detach()
                    fake_content = m(fake_content)

                    # real_vgg = norm(real_content) # instance normalize features
                    # fake_vgg = norm(fake_content)
                    real_vgg = real_content.clone()
                    fake_vgg = fake_content.clone()

                    # content_loss += [nn.L1Loss()(real_vgg, fake_vgg)]
                    content_loss += [Smooth_L1(real_vgg, fake_vgg)]
                    # content_loss += [5.*pdl_loss(real_vgg, fake_vgg, metric='charbonier', m=20)]

                    # gram matrices
                    gram_real = real_vgg.view(real_vgg.shape[0],real_vgg.shape[1],-1) @ real_vgg.view(real_vgg.shape[0],real_vgg.shape[1],-1).transpose(-2,-1)
                    gram_fake = fake_vgg.view(fake_vgg.shape[0],fake_vgg.shape[1],-1) @ fake_vgg.view(fake_vgg.shape[0],fake_vgg.shape[1],-1).transpose(-2,-1)

                    # gram_loss += [weight_list[k]*nn.L1Loss()(gram_real, gram_fake)]
                    gram_loss += [weight_list[k]*Smooth_L1(gram_real, gram_fake)]

                loss_VGG = sum(content_loss) + lambda_gram*sum(gram_loss)
                loss_VGG *= opt.lambda_vgg

                loss_G = loss_GE + loss_latent + loss_VGG
                # loss_G = loss_GE  + loss_VGG # DC has been applied

            loss_G.backward()
            optimizer_G.step()
            # optimizer_G_atasm.step()

            # scaler_G.scale_G(loss_G).backward()
            # scaler_G.step_G(optimizer_G)
            # scaler_G.update()

            # ----------------------------------
            #  Train Discriminator (cVAE-GAN)
            # ----------------------------------

            # if opt.epoch>0 and epoch == (opt.epoch+1) and i == 0:
            #         print('load optimizers here')
            #         print('load optimizers here')
            #         # Load pretrained models
            #         optimizer_D_VAE.load_state_dict(torch.load("saved_models/%s/optimizer_D_VAE_%d.pth" % (opt.dataset_name, opt.epoch)))
            #         print('load optimizers here')
            #         print('load optimizers here')

            optimizer_D_VAE.zero_grad()

            clone_B = torch.ones(fake_B.shape).cuda() # avoid issues caused by .detach()
            clone_B.copy_(fake_B)
            # clone_B = fake_B.new_tensor(fake_B)
            with torch.cuda.amp.autocast(enabled=False):
                loss_D_VAE = D_VAE.compute_loss(real_B, clone_B.detach(), valid=valid, fake=fake, sg=True) # relativistic average
                # loss_D_VAE = D_VAE.compute_loss(real_B, None, fake=valid, sg=False) + D_VAE.compute_loss(fake_B.detach(), None, fake=fake, sg=False)

                loss_D_VAE *= opt.lambda_adv 

                # gradient penalty
                loss_grad_VAE = 0.
                loss_grad_VAE = 30.*D_VAE.compute_gradient_penalty(real_B, fake_B.detach()) # gradient penalty
                loss_grad_VAE *= opt.lambda_adv

                loss_D = loss_D_VAE + loss_grad_VAE

            loss_D.backward()
            optimizer_D_VAE.step()

            # scaler_D.scale(loss_D).backward()
            # scaler_D.step(optimizer_D_VAE)
            # scaler_D.update()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[E %d/%d, %d/%d] [D: (%.3f, %.3f)] [G: (%.3f), pixel: (%.3f, %.3f, %.3f), LR: %.4f vgg: (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D_VAE.item(),
                    loss_grad_VAE,

                    loss_GE.item()-opt.lambda_pixel * loss_pixel.item(),

                    opt.lambda_pixel*(1-alpha)*L1_loss.item(),
                    opt.lambda_pixel*alpha*MS_SSIM_Loss.item(),
                    opt.lambda_pixel*0.5*loss_FA.item(),

                    loss_latent.item(),

                    opt.lambda_vgg*content_loss[0],
                    opt.lambda_vgg*content_loss[1],
                    opt.lambda_vgg*content_loss[2],

                    opt.lambda_vgg*lambda_gram*gram_loss[0],
                    opt.lambda_vgg*lambda_gram*gram_loss[1],
                    opt.lambda_vgg*lambda_gram*gram_loss[2],
                    time_left,
                )
            )

            if batches_done % opt.sample_interval == 0:
                sample_images(epoch, i)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_VAE.state_dict(), "saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, epoch))
            torch.save(optimizer_G.state_dict(), "saved_models/%s/optimizer_G_%d.pth" % (opt.dataset_name, epoch))
            torch.save(optimizer_D_VAE.state_dict(), "saved_models/%s/optimizer_D_VAE_%d.pth" % (opt.dataset_name, epoch))

            # torch.save(optimizer_G_atasm.state_dict(), "saved_models/%s/optimizer_G_atasm_%d.pth" % (opt.dataset_name, epoch))
