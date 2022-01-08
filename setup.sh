pip install -U torch pytorch_lightning timm albumentations wandb opencv-python tqdm 
wandb login

git config --global user.email "marktenenholtz@gmail.com"
git config --global user.name "Mark Tenenholtz"

mkdir data/
unzip drive/MyDrive/Kaggle/petfinder-pawpularity/petfinder-data.zip -d data/