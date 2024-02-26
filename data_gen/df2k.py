import os

from data_gen.utils import build_bin_dataset_sam, build_bin_dataset
from utils_sr.hparams import hparams, set_hparams

if __name__ == '__main__':
    set_hparams()
    
    train_img_dir = 'data/sr/DF2K/DF2K_train_HR'
    test_img_dir = 'data/sr/DIV2K/DIV2K_valid_HR'
    
    train_sam_embed_dir = 'data/sam_embed/DF2K/DF2K_train_HR'
    
    binary_data_dir = hparams['binary_data_dir']
    os.makedirs(binary_data_dir, exist_ok=True)
    
    train_img_list = sorted(os.listdir(train_img_dir))
    test_img_list = sorted(os.listdir(test_img_dir))
    
    crop_size = hparams['crop_size']
    patch_size = hparams['patch_size']
    thresh_size = hparams['thresh_size']
    test_crop_size = hparams['test_crop_size']
    test_thresh_size = hparams['test_thresh_size']
    
    build_bin_dataset_sam(train_img_list, binary_data_dir, 'train', patch_size, crop_size, thresh_size,
                          train_sam_embed_dir)
    
    build_bin_dataset(test_img_list, binary_data_dir, 'test', patch_size, crop_size, thresh_size)
