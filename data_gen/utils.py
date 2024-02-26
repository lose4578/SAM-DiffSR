import os

import cv2
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import Pool
from os import path as osp
from PIL import Image
from numpy import asarray
from tqdm import tqdm

from utils_sr.hparams import hparams
from utils_sr.indexed_datasets import IndexedDatasetBuilder
from utils_sr.matlab_resize import imresize


def worker(args):
    i, path, patch_size, crop_size, thresh_size, sr_scale = args
    img_name, extension = osp.splitext(osp.basename(path))
    img = Image.open(path).convert('RGB')
    img = asarray(img)
    h, w, c = img.shape
    h = h - h % sr_scale
    w = w - w % sr_scale
    img = img[:h, :w]
    h, w, c = img.shape
    img_lr = imresize(img, 1 / sr_scale)
    ret = []
    x = 0
    while x < h - thresh_size:
        y = 0
        while y < w - thresh_size:
            x_l_left = x // sr_scale
            x_l_right = (x + crop_size[0]) // sr_scale
            y_l_left = y // sr_scale
            y_l_right = (y + crop_size[1]) // sr_scale
            cropped_img = img[x:x + crop_size[0], y:y + crop_size[1], ...]
            cropped_img_lr = img_lr[x_l_left:x_l_right, y_l_left:y_l_right]
            ret.append({
                    'item_name': img_name,
                    'loc': [x // crop_size[0], y // crop_size[1]],
                    'loc_bdr': [(h + crop_size[0] - 1) // crop_size[0], (w + crop_size[1] - 1) // crop_size[1]],
                    'path': path, 'img': cropped_img,
                    'img_lr': cropped_img_lr,
            })
            y += crop_size[1]
        x += crop_size[0]
    
    return i, ret


def worker_sam(args):
    i, path, patch_size, crop_size, thresh_size, sr_scale, sam_dir = args
    img_name, extension = osp.splitext(osp.basename(path))
    sam_path = osp.join(sam_dir, f'{img_name}.npy')
    img = Image.open(path).convert('RGB')
    img = asarray(img)
    
    h, w, c = img.shape
    h = h - h % sr_scale
    w = w - w % sr_scale
    img = img[:h, :w]
    h, w, c = img.shape
    img_lr = imresize(img, 1 / sr_scale)
    
    try:
        sam_mask = np.load(sam_path)
    except:
        sam_mask = np.zeros(img.shape[:2])
    
    if sam_mask.shape != img.shape[:2]:
        sam_mask = cv2.resize(sam_mask, dsize=img.shape[:2][::-1])
    
    ret = []
    x = 0
    while x < h - thresh_size:
        y = 0
        while y < w - thresh_size:
            x_l_left = x // sr_scale
            x_l_right = (x + crop_size[0]) // sr_scale
            y_l_left = y // sr_scale
            y_l_right = (y + crop_size[1]) // sr_scale
            cropped_img = img[x:x + crop_size[0], y:y + crop_size[1], ...]
            cropped_img_lr = img_lr[x_l_left:x_l_right, y_l_left:y_l_right]
            cropped_sam_mask = sam_mask[x_l_left:x_l_right, y_l_left:y_l_right]
            ret.append({
                    'item_name': img_name,
                    'loc': [x // crop_size[0], y // crop_size[1]],
                    'loc_bdr': [(h + crop_size[0] - 1) // crop_size[0], (w + crop_size[1] - 1) // crop_size[1]],
                    'path': path, 'img': cropped_img,
                    'img_lr': cropped_img_lr,
                    'sam_mask': cropped_sam_mask,
                    'mask_path': sam_path
            })
            y += crop_size[1]
        x += crop_size[0]
    
    return i, ret


def build_bin_dataset(paths, binary_data_dir, prefix, patch_size, crop_size, thresh_size):
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    sr_scale = hparams['sr_scale']
    assert crop_size[0] % sr_scale == 0
    assert crop_size[1] % sr_scale == 0
    assert patch_size % sr_scale == 0
    assert thresh_size % sr_scale == 0
    
    builder = IndexedDatasetBuilder(f'{binary_data_dir}/{prefix}')
    
    def get_worker_args():
        for i, path in enumerate(paths):
            yield i, path, patch_size, crop_size, thresh_size, sr_scale
    
    with Pool(processes=10) as pool:
        for ret in tqdm(pool.imap_unordered(worker, list(get_worker_args())), total=len(paths)):
            if 'test' in prefix:
                builder.add_item(ret[1][0], id=ret[0])
            else:
                for r in ret[1]:
                    builder.add_item(r)
    builder.finalize()


def build_bin_dataset_sam(paths, binary_data_dir, prefix, patch_size, crop_size, thresh_size, sam_dir):
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    sr_scale = hparams['sr_scale']
    assert crop_size[0] % sr_scale == 0
    assert crop_size[1] % sr_scale == 0
    assert patch_size % sr_scale == 0
    assert thresh_size % sr_scale == 0
    
    builder = IndexedDatasetBuilder(f'{binary_data_dir}/{prefix}')
    
    def get_worker_args():
        for i, path in enumerate(paths):
            yield i, path, patch_size, crop_size, thresh_size, sr_scale, sam_dir
    
    with Pool(processes=20) as pool:
        for ret in tqdm(pool.imap_unordered(worker_sam, list(get_worker_args())), total=len(paths)):
            if 'test' in prefix:
                builder.add_item(ret[1][0], id=ret[0])
            else:
                for r in ret[1]:
                    builder.add_item(r)
    builder.finalize()
