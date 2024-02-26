import importlib
import os
import sys
from collections import OrderedDict
from pathlib import Path

from tasks.srdiff_df2k import InferDataSet

parent_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.abspath(parent_path))
os.chdir(parent_path)
print(f'>-------------> parent path {parent_path}')
print(f'>-------------> current work dir {os.getcwd()}')

cache_path = os.path.join(parent_path, 'cache')
os.environ["HF_DATASETS_CACHE"] = cache_path
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["torch_HOME"] = cache_path

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils_sr.hparams import hparams, set_hparams


def load_ckpt(ckpt_path, model):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    stat_dict = checkpoint['state_dict']['model']
    
    new_state_dict = OrderedDict()
    for k, v in stat_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # 去掉 `module.`
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.cuda()


def infer(trainer, ckpt_path, img_dir, save_dir):
    trainer.build_model()
    load_ckpt(ckpt_path, trainer.model)
    
    dataset = InferDataSet(img_dir)
    test_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)
    
    torch.backends.cudnn.benchmark = False
    
    with torch.no_grad():
        trainer.model.eval()
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for batch_idx, batch in pbar:
            img_lr, img_lr_up, img_name = batch
            
            img_lr = img_lr.to('cuda')
            img_lr_up = img_lr_up.to('cuda')
            
            img_sr, _ = trainer.model.sample(img_lr, img_lr_up, img_lr_up.shape)
            
            img_sr = img_sr.clamp(-1, 1)
            img_sr = trainer.tensor2img(img_sr)[0]
            img_sr = Image.fromarray(img_sr)
            img_sr.save(os.path.join(save_dir, img_name[0]))


if __name__ == '__main__':
    set_hparams()
    
    img_dir = hparams['img_dir']
    save_dir = hparams['save_dir']
    ckpt_path = hparams['ckpt_path']
    
    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    cls_name = hparams["trainer_cls"].split(".")[-1]
    trainer = getattr(importlib.import_module(pkg), cls_name)()
    
    os.makedirs(save_dir, exist_ok=True)
    
    infer(trainer, ckpt_path, img_dir, save_dir)
