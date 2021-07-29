#python '/home/liu/Github_Projects/fastMRI_data/h5topng/utils.py'
import os
import argparse
import h5py
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
#import cv2

args=argparse.ArgumentParser()

args.add_argument('--resize', type=int, nargs='+', default=(512,256))

args=args.parse_args()

#return k-space in h5: (Batches,Channels, H, W)
def load_single(path):
    hf = h5py.File(path)
    volume_kspace = hf['kspace'][()]
    return np.expand_dims(volume_kspace,axis=1)

def load_multiple(path):
    hf = h5py.File(path)
    return hf['kspace'][()]
        
#reconstruction
from data import transforms as T

def kspace_T_multiple(volume_kspace,resize):
    #retun reconstructions-(B,C=1,H, W) and coil frames=(B,C=15,H,W)
    
    #show in k-space some coil channels of one image
#    show_slices(np.log(np.abs(volume_kspace[10]) + 1e-9), [0, 5, 10],cmap='gray', name='k_space')
    
    volume_kspace = T.to_tensor(volume_kspace)      # Convert from numpy array to pytorch tensor
    volume_kspace=volume_kspace.to(device)
    volume_image = T.ifft2(volume_kspace)           # Apply Inverse Fourier Transform to get the complex image
    volume_image = T.complex_abs(volume_image)   # Compute absolute value to get a real image
    
    #Multi-coil Reconstion: root sum of square
    volume_image_rss = T.root_sum_of_squares(volume_image, dim=1)
    volume_image_rss = torch.unsqueeze(volume_image_rss, dim=1)
#    volume_image_rss=scale(volume_image_rss)
    
    #resize and scale
    if resize != None:
        volume_image=tensor_resize(volume_image,resize)
        volume_image_rss=tensor_resize(volume_image_rss,resize)
    else:
        pass
    #show in image space some coil channels of one image
#    show_slices(volume_image[10], [0, 5, 10], cmap='gray',name='image_space')
    
#    fig=plt.figure(figsize=(8,6))
#    plt.imshow(np.abs(volume_image_rss[10].cpu().data.numpy()), cmap='gray')
#    fig.show()
#    plt.pause(1)
#    fig.savefig('root_sum_of_squares',format='png')
    
    return volume_image_rss.cpu().data.numpy(), volume_image.cpu().data.numpy()
    
def kspace_T_single(volume_kspace,resize=None):
    #retun reconstructions-(B,C=1,H, W)
    
    #show in k-space some coil channels of one image
#    show_slices(np.log(np.abs(volume_kspace[10]) + 1e-9), [0],cmap='gray',name='k_space')
    
    volume_kspace = T.to_tensor(volume_kspace)     # Convert from numpy array to pytorch tensor
    volume_kspace=volume_kspace.to(device)
    volume_image = T.ifft2(volume_kspace)           # Apply Inverse Fourier Transform to get the complex image
    volume_image = T.complex_abs(volume_image)   # Compute absolute value to get a real image
    
    #resize and scale
    if resize != None:
        volume_image=tensor_resize(volume_image,resize)
    else:
        pass
    #show in image space some coil channels of one image
#    show_slices(volume_image[10], [0], cmap='gray',name='image_space')
    
    return volume_image.cpu().data.numpy()
      
#plot
def show_slices(data, slice_nums, cmap='gray', name=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
    fig.show()
    plt.pause(1)
    if name !=None:
        fig.savefig(name,format='png')
        
#scale
def scale(frame,scale=None):
    if scale ==None:
        return (frame-frame.min())/(frame.max()-frame.min())
    else:
        return frame*scale

#resize
def tensor_resize(data,resize_shape):
    shape=data.shape
    resized=torch.empty((shape[0],shape[1],resize_shape[0],resize_shape[1]))
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            frame=data[i][j]
            frame=scale(frame) #scale before resize
            
            img=transforms.ToPILImage()(frame.cpu()) #convert to PIL Image
            img=transforms.Resize(resize_shape)(img)#resize
            
            resized[i][j]=transforms.ToTensor()(img)

    return scale(resized)


######################################################################################
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.backends.cudnn.enabled)

#test

folder = '../../fastMRI_data/NYU_fastMRI/single_train'#h5 folder

try:
    os.mkdir('../../fastMRI_data/NYU_fastMRI/single_train_png/')
except OSError:
    pass

filelist=os.listdir(folder)# list of h5 filenames

for filename in filelist:
#    filename=os.path.splitext(filename)
    path=os.path.join(folder,filename)
    
    volume_kspace = load_single(path)
    volume_image=kspace_T_single(volume_kspace, args.resize)
    
#    print(volume_kspace.shape,volume_image.shape)
#    print(volume_kspace.dtype,volume_image.dtype)

    #create a folder for each h5 file
    frame_folder=os.path.join('../../fastMRI_data/NYU_fastMRI/single_train_png/',os.path.splitext(filename)[0])
    try:
        os.mkdir(frame_folder)
    except OSError:
        pass
    
    for i in range(volume_image.shape[0]):
        for j in range(volume_image.shape[1]):
            frame_name='B_%d_C_%d.png' % (i, j)
            frame_save_path = os.path.join(frame_folder, frame_name)
            plt.imsave(frame_save_path, volume_image[i][j], format='png', cmap='gray')


#################################################################################################
#folder = '../../fastMRI_data/NYU_fastMRI/multi_train'#h5 folder

#try:
#    os.mkdir('../../fastMRI_data/NYU_fastMRI/multi_train_png/')
#except OSError:
#    pass

#filelist=os.listdir(folder)# list of h5 filenames

#for filename in filelist:
##    filename=os.path.splitext(filename)
#    path=os.path.join(folder,filename)
#    
#    volume_kspace = load_multiple(path)
#    volume_image_rss, volume_image_coil =kspace_T_multiple(volume_kspace,resize=args.resize)
#    
##    print(volume_kspace.shape,volume_image.shape)
##    print(volume_kspace.dtype,volume_image.dtype)

#    #create a folder for each h5 file
#    frame_folder=os.path.join('../../fastMRI_data/NYU_fastMRI/multi_train_png/', os.path.splitext(filename)[0])
#    try:
#        os.mkdir(frame_folder)
#    except OSError:
#        pass
#        
#    #create folders for rss and coil frames
#    frame_folder_rss=os.path.join(frame_folder,'rss')
#    frame_folder_coil=os.path.join(frame_folder,'coil')
#    try:
#        os.mkdir(frame_folder_rss)
#        os.mkdir(frame_folder_coil)
#    except OSError:
#        pass
#    
#    
#    for i in range(volume_image_rss.shape[0]):
#        for j in range(volume_image_rss.shape[1]):
#            frame_name='B_%d_C_%d.png' % (i, j)
#            
#            frame_save_path = os.path.join(frame_folder_rss, frame_name)
#            plt.imsave(frame_save_path, volume_image_rss[i][j], format='png', cmap='gray')

#    for i in range(volume_image_coil.shape[0]):
#        for j in range(volume_image_coil.shape[1]):
#            frame_name='B_%d_C_%d.png' % (i, j)

#            frame_save_path = os.path.join(frame_folder_coil, frame_name)         
##            plt.imsave(frame_save_path, volume_image[i][j], format='png', cmap='gray')


