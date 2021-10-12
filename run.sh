python train.py --model_name swin_base_patch4_window12_384_in22k --name random_resize --lr 1e-5 --wd 0. --batch_size 4 --accumulate_grad_batches 8 --img_size_x 384 --img_size_y 384
python train.py --model_name xcit_medium_24_p8_224_dist --name random_resize --lr 1e-5 --wd 0. --batch_size 4 --accumulate_grad_batches 8 --img_size_x 224 --img_size_y 224
python train.py --model_name eca_nfnet_l2 --name random_resize --lr 1e-5 --wd 0. --batch_size 4 --accumulate_grad_batches 8 --img_size_x 512 --img_size_y 512
#python train.py --model_name swin_large_patch4_window7_224_in22k --name random_resize --lr 1e-5 --wd 0 --batch_size 8 --accumulate_grad_batches 4