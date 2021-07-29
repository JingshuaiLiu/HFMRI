
#
python multicoil_brain_baseline.py --n_epochs 1 --lr 0.0005 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 100 --checkpoint_interval 1 --epoch 35 --mask ../brain_dataset/brain_mask_0_125.pt --dataroot ../brain_dataset/train --testroot ../brain_dataset/test --lambda_adv 0.01 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.1 --dataset_name brain_database/multi_coil_dense_network_stasm_Contextual --stasm --data_consistency

python multicoil_brain_analysis.py --n_epochs 1 --lr 0.0005 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 100 --checkpoint_interval 1 --epoch 34 --mask ../brain_dataset/brain_mask_0_125.pt --dataroot ../brain_dataset/train --testroot ../brain_dataset/test --lambda_adv 0.01 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.1 --dataset_name brain_database/multi_coil_dense_network_stasm_Contextual --stasm --data_consistency

python fid_score.py --true ./outputs/output_dataset_fully --fake ./outputs/output_dataset_recons --batch-size 10 --gpu 0

python kid_score.py --true ./outputs/output_dataset_fully --fake ./outputs/output_dataset_recons --batch-size 10 --gpu 0

#
python multicoil_brain_baseline.py --n_epochs 1 --lr 0.0005 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 100 --checkpoint_interval 1 --epoch 34 --mask ../brain_dataset/brain_mask_0_125.pt --dataroot ../brain_dataset/train --testroot ../brain_dataset/test --lambda_adv 0.01 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.1 --dataset_name brain_database/multi_coil_dense_network_stasm_Transformer --stasm --data_consistency

python multicoil_brain_analysis.py --n_epochs 1 --lr 0.0005 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 100 --checkpoint_interval 1 --epoch 34 --mask ../brain_dataset/brain_mask_0_125.pt --dataroot ../brain_dataset/train --testroot ../brain_dataset/test --lambda_adv 0.01 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.1 --dataset_name brain_database/multi_coil_dense_network_stasm_Transformer --stasm --data_consistency

#
python multicoil_brain_baseline.py --n_epochs 1 --lr 0.0005 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 100 --checkpoint_interval 1 --epoch 36 --mask ../brain_dataset/brain_mask_0_125.pt --dataroot ../brain_dataset/train --testroot ../brain_dataset/test --lambda_adv 0.01 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.1 --dataset_name brain_database/multi_coil_dense_network_seprepmlp_stasm_Contextual --stasm --data_consistency

python multicoil_brain_analysis.py --n_epochs 1 --lr 0.0005 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 100 --checkpoint_interval 1 --epoch 37 --mask ../brain_dataset/brain_mask_0_125.pt --dataroot ../brain_dataset/train --testroot ../brain_dataset/test --lambda_adv 0.01 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.1 --dataset_name brain_database/multi_coil_dense_network_seprepmlp_stasm_Contextual --stasm --data_consistency

E36-29
 * Average PSNR 28.0890
 * Average SSIM 0.8529
 * Average FID 81.27
 * Average KID 0.039

 * Average PSNR 27.9105
 * Average SSIM 0.8487
 * Average FID 78.55
 * Average KID 0.039

 * Average PSNR 27.8412
 * Average SSIM 0.8459

 * Average PSNR 27.9029
 * Average SSIM 0.8473

 * Average PSNR 27.3963
 * Average SSIM 0.8348

 * Average PSNR 27.9970
 * Average SSIM 0.8492

 * Average PSNR 27.7623
 * Average SSIM 0.8428

 * Average PSNR 27.8588
 * Average SSIM 0.8463

    ''' '''
python multicoil_brain_baseline.py --n_epochs 5 --lr 0.0005 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 100 --checkpoint_interval 1 --epoch 30 --mask ../brain_dataset/brain_mask_0_125.pt --dataroot ../brain_dataset/train --testroot ../brain_dataset/test --lambda_adv 0.01 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.1 --dataset_name brain_database/multi_coil_dense_network_stasm_Contextual_hierarchi --stasm --data_consistency

python multicoil_brain_analysis.py --n_epochs 1 --lr 0.0005 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 100 --checkpoint_interval 1 --epoch 34 --mask ../brain_dataset/brain_mask_0_125.pt --dataroot ../brain_dataset/train --testroot ../brain_dataset/test --lambda_adv 0.01 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.1 --dataset_name brain_database/multi_coil_dense_network_stasm_Contextual_hierarchi --stasm --data_consistency

 * Average PSNR 27.7067
 * Average SSIM 0.8449

 * Average PSNR 27.8025
 * Average SSIM 0.8469

  * Average PSNR 27.7480
 * Average SSIM 0.8438

  * Average PSNR 27.7066
 * Average SSIM 0.8443

  * Average PSNR 27.7282
 * Average SSIM 0.8456

import torch
import torchvision.utils as vtils
names = ['attention_maps_1.pt','attention_maps_2.pt','attention_maps_3.pt']
bar = torch.ones((*img.shape[:-1], 15))
img = []
for n in names:
    M = torch.load(n).unsqueeze(1)[[0]]
    # M = M.transpose(-1,-2)
    img.append(M.repeat_interleave(repeats=15,dim=-2).repeat_interleave(repeats=15,dim=-1))
img = torch.cat([img[0], bar, img[1], bar, img[2]], dim=-1)
vtils.save_image(img, 'attention_maps_C.png')


''' single coil contextual channel attention, --lambda_adv(0.1)/2 after 30 epochs '''
python single_coil_dense_network.py --n_epochs 1 --lr 0.00001 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 500 --checkpoint_interval 1 --epoch 50 --mask ../cyclegan_MRI/data/mask_0_125.pt --dataroot ../cyclegan_MRI/data/include_all_coils_training_12_slices_norm.pt --lambda_adv 0.05 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.5 --dataset_name single_coil_dense_network_stasm_dc_contextual --stasm --data_consistency

E50-49
 * Average PSNR 26.9067(1.69393313) # E50, / .max, h=0.25
 * Average SSIM 0.6828(0.06228448)
 * FID 81.50 (2.125)
 * KID 0.017 (0.001)

 * Average PSNR 26.9347
 * Average SSIM 0.6831
 * Average FID 85.14
 * Average KID 0.016

E55-51
 * Average PSNR 26.9041
 * Average SSIM 0.6720 

 * Average PSNR 26.9618
 * Average SSIM 0.6755

 * Average PSNR 26.9408
 * Average SSIM 0.6809

 * Average PSNR 26.8785
 * Average SSIM 0.6757

 * Average PSNR 26.9299
 * Average SSIM 0.6765

 ''' single coil contextual channel attention and non-local spatial attention selection (using pixel shift, contextual distance, and direction vectors to adaptively predict the kernels), --lambda_adv(0.1)/2 after 30 epochs '''

python single_coil_dense_network.py --n_epochs 1 --lr 0.00001 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 500 --checkpoint_interval 1 --epoch 56 --mask ../cyclegan_MRI/data/mask_0_125.pt --dataroot ../cyclegan_MRI/data/include_all_coils_training_12_slices_norm.pt --lambda_adv 0.05 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.5 --dataset_name single_coil_dense_network_seprepmlp_stasm_dc_contextual --stasm --data_consistency

E56-50      
 * Average PSNR 26.9660(1.67121543)
 * Average SSIM 0.6800(0.06515470)
 * Average FID 80.10 (1.154)
 * Average KID 0.017 (0.001)

 * Average PSNR 26.9677(1.65565264) # E55 / .max, h=0.25
 * Average SSIM 0.6931(0.06313463)
 * Average FID  77.71 (2.096)
 * Average KID 0.016 (0.001)

 * Average PSNR 26.9853 # E55, / .min, h=0.2
 * Average SSIM 0.6847
 * Average FID  77.57 / 79.23
 * Average KID 0.015

 * Average PSNR 26.9137
 * Average SSIM 0.6808

 * Average PSNR 26.9753
 * Average SSIM 0.6835
 * Average FID 81.28
 * Average KID 0.011

 * Average PSNR 27.0455(1.66617844) # E52
 * Average SSIM 0.6983(0.06262811)
 * Average FID 82.31 (1.596)
 * Average KID 0.019 (0.001)

 * Average PSNR 26.9226
 * Average SSIM 0.6836
 * Average FID 82.24
 * Average KID 0.013

 * Average PSNR 26.9391
 * Average SSIM 0.6837
 * Average FID 77.59 / 82.83
 * Average KID 0.014 / 0.013

  ''' single coil contextual attention and hierarchical spatial attention selection, --lambda_adv(0.1)/2 after 30 epochs '''
python single_coil_dense_network.py --n_epochs 5 --lr 0.00001 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 500 --checkpoint_interval 1 --epoch 5 --mask ../cyclegan_MRI/data/mask_0_125.pt --dataroot ../cyclegan_MRI/data/include_all_coils_training_12_slices_norm.pt --lambda_adv 0.1 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.5 --dataset_name single_coil_dense_network_stasm_dc_context_hierarchi --stasm --data_consistency


python multicoil_brain_analysis.py --n_epochs 1 --lr 0.0005 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 100 --checkpoint_interval 1 --epoch 33 --mask ../brain_dataset/brain_mask_0_125.pt --dataroot ../brain_dataset/train --testroot ../brain_dataset/test --lambda_adv 0.01 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.1 --dataset_name brain_database/multi_coil_dense_network_stasm_2 --stasm --data_consistency