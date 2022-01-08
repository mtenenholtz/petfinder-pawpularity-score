# python validate.py --model_name eca_nfnet_l2-seed-4242-random_crop --img_size_x 512 --img_size_y 512 --batch_size 8 --data_seed 42
# python validate.py --model_name eca_nfnet_l2-seed-2626-random_crop --img_size_x 512 --img_size_y 512 --batch_size 8 --data_seed 26

python validate.py --model_name swin_base_patch4_window12_384_in22k-seed-4242-random_resize --img_size_x 384 --img_size_y 384 --batch_size 8 --data_seed 42
python validate.py --model_name swin_base_patch4_window12_384_in22k-seed-2626-random_resize --img_size_x 384 --img_size_y 384 --batch_size 8 --data_seed 26

python validate.py --model_name swin_large_patch4_window7_224_in22k-seed-4242-random_resize --batch_size 8 --data_seed 42
python validate.py --model_name swin_large_patch4_window7_224_in22k-seed-2626-random_resize --batch_size 8 --data_seed 26