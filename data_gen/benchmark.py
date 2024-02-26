import os

os.environ["OMP_NUM_THREADS"] = "1"

from data_gen.utils import build_bin_dataset
from utils_sr.hparams import hparams, set_hparams

if __name__ == '__main__':
    data_name = 'Set14'     # Set5 Set14 Urban100 Manga109 BSDS100
    data_path = 'data/benchmark/Set14/HR'
    
    set_hparams()
    binary_data_dir = hparams['binary_data_dir']
    os.makedirs(binary_data_dir, exist_ok=True)
    
    test_img_list = sorted(os.listdir(data_path))
    
    crop_size = hparams['crop_size']
    patch_size = hparams['patch_size']
    thresh_size = hparams['thresh_size']
    test_crop_size = hparams['test_crop_size']
    test_thresh_size = hparams['test_thresh_size']
    
    build_bin_dataset(test_img_list, binary_data_dir, f'test_{data_name}', patch_size, test_crop_size, test_thresh_size)
