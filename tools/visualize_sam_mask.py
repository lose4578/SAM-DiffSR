import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

num = '0824'

sam_npy = '/home/ma-user/work/data/sr_sam/merge_RoPE/DF2K/DF2K_train_HR'
save_dir = '/home/ma-user/work/data/sr_sam/merge_RoPE/vis/DF2K/DF2K_train_HR'

os.makedirs(save_dir, exist_ok=True)

for file in tqdm(glob.glob(f'{sam_npy}/*.npy')):
    name = os.path.basename(file).split('.')[0]
    save_path = os.path.join(save_dir, f'{name}.png')
    img = np.load(file)
    plt.imshow(img)
    plt.savefig(save_path)
