# python train.py --model_name eca_nfnet_l2 --name random_crop --lr 2e-5 --wd 0. --batch_size 3 --accumulate_grad_batches 9 --img_size_x 512 --img_size_y 512 --seed 42 --fold 2
# python train.py --model_name eca_nfnet_l2 --name random_crop --lr 2e-5 --wd 0. --batch_size 3 --accumulate_grad_batches 9 --img_size_x 512 --img_size_y 512 --seed 42 --fold 3
# python train.py --model_name eca_nfnet_l2 --name random_crop --lr 2e-5 --wd 0. --batch_size 3 --accumulate_grad_batches 9 --img_size_x 512 --img_size_y 512 --seed 42 --fold 4
# python train.py --model_name eca_nfnet_l2 --name random_crop --lr 2e-5 --wd 0. --batch_size 3 --accumulate_grad_batches 9 --img_size_x 512 --img_size_y 512 --seed 26

# python train.py --model_name swin_base_patch4_window12_384_in22k --name random_resize --lr 2e-5 --wd 1e-2 --batch_size 4 --accumulate_grad_batches 8 --img_size_x 384 --img_size_y 384 --seed 42
# python train.py --model_name swin_base_patch4_window12_384_in22k --name random_resize --lr 2e-5 --wd 1e-2 --batch_size 4 --accumulate_grad_batches 8 --img_size_x 384 --img_size_y 384 --seed 26

#python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --fold 9
# python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 42 --fold 3
# python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 42 --fold 4
# python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 42 --fold 5
# python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 42 --fold 6
# python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 42 --fold 7
# python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 42 --fold 8
# python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 42 --fold 9
# python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 26

# python train.py --model_name convit_base --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 16 --accumulate_grad_batches 2
# python train.py --model_name convit_base --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 16 --accumulate_grad_batches 2 --seed 42
# python train.py --model_name convit_base --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 16 --accumulate_grad_batches 2 --seed 26

python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32
python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 42
python train.py --model_name convit_small --name random_resize_reg_aug --lr 2e-5 --wd 0.1 --batch_size 32 --seed 26