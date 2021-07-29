python single_coil_dense_network.py --n_epochs 50 --lr 0.00001 --b1 0.5 --b2 0.999 --batch_size 1 --sample_interval 500 --checkpoint_interval 1 --epoch 0 --mask path/to/mask --dataroot path/to/training_dataset.pt --lambda_adv 0.05 --lambda_pixel 10. --lambda_latent 0.5 --lambda_vgg 0.5 --dataset_name save_data_name --stasm --data_consistency

