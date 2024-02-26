import os
import ssl
from os.path import join
from pathlib import Path
from statistics import mean

parent_path = Path(__file__).absolute().parent.parent
parent_path = os.path.abspath(parent_path)

os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

cache_path = os.path.join(parent_path, 'cache')
os.environ["HF_DATASETS_CACHE"] = cache_path
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["torch_HOME"] = cache_path

import PIL
import numpy as np
import pandas as pd
import pyiqa
import torch
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

metric_dict = {
        'psnr-Y': pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr'),
        'ssim': pyiqa.create_metric('ssim', color_space='ycbcr'),
        'fid': pyiqa.create_metric('fid'),
}


def load_img(path, target_size=None):
    image = Image.open(path).convert("RGB")
    if target_size:
        h, w = target_size
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def eval_img_IQA(gt_dir, sr_dir, excel_path, metric_list, exp_name, data_name):
    gt_img_list = os.listdir(gt_dir)
    
    iqa_result = {}
    
    for metric in metric_list:
        iqa_metric = metric_dict[metric].to(device)
        score_fr_list = []
        
        if metric == 'fid':
            score_fr = iqa_metric(sr_dir, gt_dir)
            iqa_result[metric] = float(score_fr)
            print(f'{metric}: {float(score_fr)}')
        else:
            for img_name in tqdm(gt_img_list):
                base_name = img_name.split('.')[0]
                sr_img_name = f'{base_name}.png'
                gt_img_path = join(gt_dir, img_name)
                sr_img_path = join(sr_dir, sr_img_name)
                
                if not os.path.exists(sr_img_path):
                    print(f'File not exist: {sr_img_path}')
                    continue
                
                gt_img = load_img(gt_img_path, target_size=None)
                target_size = gt_img.shape[2:]
                sr_img = load_img(sr_img_path, target_size=target_size)
                
                score_fr = iqa_metric(sr_img, gt_img)
                
                if score_fr.shape == (1,):
                    score_fr = score_fr[0]
                    if isinstance(score_fr, torch.Tensor):
                        score_fr = float(score_fr.cpu().numpy())
                else:
                    score_fr = float(score_fr)
                score_fr_list.append(score_fr)
            
            mean_score = mean(score_fr_list)
            iqa_result[metric] = float(mean_score)
            print(f'{metric}: {mean_score}')
    
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        df = pd.DataFrame(columns=['exp'])
    
    new_index = len(df.index)
    
    exp_name = int(exp_name)
    if exp_name in df['exp'].to_list():
        new_index = df[df['exp'] == exp_name].index.tolist()[0]
    else:
        df.loc[new_index, 'exp'] = exp_name
    
    for index, metric in enumerate(metric_list):
        df_metric = f'{data_name}-{metric}'
        if df_metric not in df.columns.tolist():
            df[df_metric] = ''
        
        df.loc[new_index, df_metric] = iqa_result[metric]
    
    df.sort_values(by='exp', inplace=True)
    
    df.to_excel(excel_path, startcol=0, index=False)


def main():
    epoch = 400000
    add_name = ''
    exp_root = '/home/ma-user/work/code/SRDiff-main/checkpoints'
    
    model_type_list = ['diffsr_df2k4x_sam-pl_qs-zero']
    
    metric_list = ['psnr-Y', 'ssim', 'fid']
    benchmark_name_list = ['test_Set5', 'test_Set14', 'test_Urban100', 'test_Manga109', 'test_BSDS100']
    
    # if benchmark:
    for model_type in model_type_list:
        excel_path = join(exp_root, model_type, f'IQA-val-{model_type}.xls')
        for benchmark_name in benchmark_name_list:
            exp_dir = join(exp_root, f'{model_type}/results_{epoch}_{add_name}/benchmark/{benchmark_name}')
            gt_img_dir = join(exp_dir, 'HR')
            sr_img_dir = join(exp_dir, 'SR')
            
            data_name = benchmark_name[5:]
            eval_img_IQA(gt_img_dir, sr_img_dir, excel_path, metric_list, epoch, data_name)


if __name__ == '__main__':
    main()
