import argparse
import json
import os
import sys
import threading
from pathlib import Path

parent_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.abspath(parent_path))
os.chdir(parent_path)

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from rotary_embedding_torch import RotaryEmbedding
from pycocotools import mask as mask_utils

NUM_THREADS = threading.active_count()


def merge_all_mask_to_one(all_mask):
    w, h = all_mask[0].shape
    one_mask = np.ones(w * h)
    merge_mask = np.zeros((w, h))
    
    for mask in all_mask:
        int_mask = mask.reshape(w * h)
        cosine_emmbed = one_mask.dot(int_mask) / (np.linalg.norm(one_mask) * np.linalg.norm(int_mask))
        merge_mask[mask] = cosine_emmbed
    
    return merge_mask


def merge_all_mask_to_one_RoPE(all_mask):
    w, h = all_mask[0].shape
    merge_mask = torch.zeros((w, h))
    
    rotary_emb = RotaryEmbedding(dim=h)
    mask_embed_ori = torch.ones(1, 1, w, h)  # queries - (batch, heads, seq len, dimension of head)
    mask_embed_ori = rotary_emb.rotate_queries_or_keys(mask_embed_ori)
    
    for mask in all_mask:
        mask = torch.tensor(mask).unsqueeze(dim=0).unsqueeze(dim=0)
        mask_embed_num = (mask_embed_ori * mask).mean()
        mask_embed = torch.ones(1, 1, w, h) * mask_embed_num
        merge_mask = torch.where(mask.bool(), mask_embed, merge_mask)
    
    merge_mask = merge_mask[0, 0, ...].numpy()
    
    return merge_mask


def merge_all_mask_to_one_sincos(all_mask):
    w, h = all_mask[0].shape
    merge_mask = np.zeros((w, h))
    
    vct = np.array([[1 / np.power(0.001, 2 * i / w) for i in range(w)]])
    row_sin = np.sin(vct)
    col_cos = np.cos(row_sin.T)
    mask_embed_ori = row_sin * col_cos
    
    for mask in all_mask:
        mask_embed_num = (mask_embed_ori * mask).mean()
        merge_mask[mask] = mask_embed_num
    
    return merge_mask


def merge_all_mask_to_one_linear(all_mask):
    w, h = all_mask[0].shape
    merge_mask = np.zeros((w, h))
    
    row_emb = np.random.uniform(0, 1, (1, w))
    col_emb = np.random.uniform(0, 1, (h, 1))
    
    mask_embed_ori = row_emb * col_emb
    
    for mask in all_mask:
        mask_embed_num = (mask_embed_ori * mask).mean()
        merge_mask[mask] = mask_embed_num
    
    return merge_mask


def merge_masks(mask_path, output_dir):
    base_name = os.path.basename(mask_path).split('.')[0]
    output_path = os.path.join(output_dir, f'{base_name}.npy')
    
    with open(mask_path) as f:
        annotation = json.load(f)
    
    mask_list = []
    for ann in annotation:
        mask = mask_utils.decode(ann["segmentation"])
        mask_list.append(mask)
    
    if len(mask_list) == 0:
        sam_mask = None
    else:
        sam_mask = merge_all_mask_to_one_RoPE(mask_list)
    
    np.save(output_path, sam_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, default='', help='input RLE format sam mask data path')
    parser.add_argument('--output', type=str, default='', help='output embedding mask data path')
    args = parser.parse_args()
    
    print(f"Processing {args.input}...")
    targets = list(Path(args.input).glob('*.json'))
    os.makedirs(args.output, exist_ok=True)
    
    Parallel(n_jobs=NUM_THREADS)(delayed(merge_masks)(t, args.output) for t in tqdm(targets))
