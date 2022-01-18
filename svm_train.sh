python svm_train.py --model_name swin_base_patch4_window12_384_in22k-seed-26-random_resize_reg_aug_ten_fold_pseudo --batch_size 8 --img_size_x 384 --img_size_y 384
python svm_train.py --model_name swin_base_patch4_window12_384_in22k-seed-34-random_resize_reg_aug_ten_fold_pseudo --batch_size 8 --img_size_x 384 --img_size_y 384
python svm_train.py --model_name swin_base_patch4_window12_384_in22k-seed-42-random_resize_reg_aug_ten_fold_pseudo --batch_size 8 --img_size_x 384 --img_size_y 384

python svm_train.py --model_name swin_large_patch4_window7_224_in22k-seed-34-random_resize_reg_aug_ten_fold_pseudo --batch_size 8
python svm_train.py --model_name swin_large_patch4_window7_224_in22k-seed-26-random_resize_reg_aug_ten_fold_pseudo --batch_size 8
python svm_train.py --model_name swin_large_patch4_window7_224_in22k-seed-42-random_resize_reg_aug_ten_fold_pseudo --batch_size 8

python svm_train.py --model_name convit_small-seed-34-random_resize_reg_aug_ten_fold_pseudo --batch_size 32
python svm_train.py --model_name convit_small-seed-42-random_resize_reg_aug_ten_fold_pseudo --batch_size 32