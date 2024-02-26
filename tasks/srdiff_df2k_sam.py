import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from rotary_embedding_torch import RotaryEmbedding
from torchvision import transforms

from models_sr.diffsr_modules import RRDBNet, Unet
from models_sr.diffusion_sam import GaussianDiffusion_sam
from tasks.srdiff import SRDiffTrainer
from utils_sr.dataset import SRDataSet
from utils_sr.hparams import hparams
from utils_sr.indexed_datasets import IndexedDataset
from utils_sr.matlab_resize import imresize
from utils_sr.utils import load_ckpt


def normalize_01(data):
    mu = np.mean(data)
    sigma = np.std(data)
    
    if sigma == 0.:
        return data - mu
    else:
        return (data - mu) / sigma


def normalize_11(data):
    mu = np.mean(data)
    sigma = np.std(data)
    
    if sigma == 0.:
        return data - mu
    else:
        return (data - mu) / sigma - 1


class Df2kDataSet_sam(SRDataSet):
    def __init__(self, prefix='train'):
        
        if prefix == 'valid':
            _prefix = 'test'
        else:
            _prefix = prefix
        
        super().__init__(_prefix)
        
        self.patch_size = hparams['patch_size']
        self.patch_size_lr = hparams['patch_size'] // hparams['sr_scale']
        if prefix == 'valid':
            self.len = hparams['eval_batch_size'] * hparams['valid_steps']
        
        self.data_position_aug_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20, interpolation=Image.BICUBIC),
        ])
        
        self.data_color_aug_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
        
        self.sam_config = hparams.get('sam_config', False)
        
        if self.sam_config.get('mask_RoPE', False):
            h, w = map(int, self.sam_config['mask_RoPE_shape'].split('-'))
            rotary_emb = RotaryEmbedding(dim=h)
            sam_mask = rotary_emb.rotate_queries_or_keys(torch.ones(1, 1, w, h))
            self.RoPE_mask = sam_mask.cpu().numpy()[0, 0, ...]
    
    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]
    
    def __getitem__(self, index):
        item = self._get_item(index)
        hparams = self.hparams
        sr_scale = hparams['sr_scale']
        
        img_hr = np.uint8(item['img'])
        img_lr = np.uint8(item['img_lr'])
        
        if self.sam_config.get('mask_RoPE', False):
            sam_mask = self.RoPE_mask
        else:
            if 'sam_mask' in item:
                sam_mask = item['sam_mask']
                if sam_mask.shape != img_hr.shape[:2]:
                    sam_mask = cv2.resize(sam_mask, dsize=img_hr.shape[:2][::-1])
            else:
                sam_mask = np.zeros_like(img_lr)
        
        # TODO: clip for SRFlow
        h, w, c = img_hr.shape
        h = h - h % (sr_scale * 2)
        w = w - w % (sr_scale * 2)
        h_l = h // sr_scale
        w_l = w // sr_scale
        img_hr = img_hr[:h, :w]
        sam_mask = sam_mask[:h, :w]
        img_lr = img_lr[:h_l, :w_l]
        
        # random crop
        if self.prefix == 'train':
            if self.data_augmentation and random.random() < 0.5:
                img_hr, img_lr, sam_mask = self.data_augment(img_hr, img_lr, sam_mask)
            i = random.randint(0, h - self.patch_size) // sr_scale * sr_scale
            i_lr = i // sr_scale
            j = random.randint(0, w - self.patch_size) // sr_scale * sr_scale
            j_lr = j // sr_scale
            img_hr = img_hr[i:i + self.patch_size, j:j + self.patch_size]
            sam_mask = sam_mask[i:i + self.patch_size, j:j + self.patch_size]
            img_lr = img_lr[i_lr:i_lr + self.patch_size_lr, j_lr:j_lr + self.patch_size_lr]
        
        img_lr_up = imresize(img_lr / 256, hparams['sr_scale'])  # np.float [H, W, C]
        img_hr, img_lr, img_lr_up = [self.to_tensor_norm(x).float() for x in [img_hr, img_lr, img_lr_up]]
        
        if hparams['sam_data_config']['all_same_mask_to_zero']:
            if len(np.unique(sam_mask)) == 1:
                sam_mask = np.zeros_like(sam_mask)
        
        if hparams['sam_data_config']['normalize_01']:
            if len(np.unique(sam_mask)) != 1:
                sam_mask = normalize_01(sam_mask)
        
        if hparams['sam_data_config']['normalize_11']:
            if len(np.unique(sam_mask)) != 1:
                sam_mask = normalize_11(sam_mask)
        
        sam_mask = torch.FloatTensor(sam_mask).unsqueeze(dim=0)
        
        return {
                'img_hr': img_hr, 'img_lr': img_lr,
                'img_lr_up': img_lr_up, 'item_name': item['item_name'],
                'loc': np.array(item['loc']), 'loc_bdr': np.array(item['loc_bdr']),
                'sam_mask': sam_mask
        }
    
    def __len__(self):
        return self.len
    
    def data_augment(self, img_hr, img_lr, sam_mask):
        sr_scale = self.hparams['sr_scale']
        img_hr = Image.fromarray(img_hr)
        img_hr, sam_mask = self.data_position_aug_transforms([img_hr, sam_mask])
        img_hr = self.data_color_aug_transforms(img_hr)
        img_hr = np.asarray(img_hr)  # np.uint8 [H, W, C]
        img_lr = imresize(img_hr, 1 / sr_scale)
        return img_hr, img_lr, sam_mask


class SRDiffDf2k_sam(SRDiffTrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = Df2kDataSet_sam
        self.sam_config = hparams['sam_config']
    
    def build_model(self):
        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        dim_mults = [int(x) for x in dim_mults.split('|')]
        
        denoise_fn = Unet(
                hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
        if hparams['use_rrdb']:
            rrdb = RRDBNet(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                           hparams['rrdb_num_feat'] // 2)
            if hparams['rrdb_ckpt'] != '' and os.path.exists(hparams['rrdb_ckpt']):
                load_ckpt(rrdb, hparams['rrdb_ckpt'])
        else:
            rrdb = None
        self.model = GaussianDiffusion_sam(
                denoise_fn=denoise_fn,
                rrdb_net=rrdb,
                timesteps=hparams['timesteps'],
                loss_type=hparams['loss_type'],
                sam_config=hparams['sam_config']
        )
        self.global_step = 0
        return self.model
    
    # def sample_and_test(self, sample):
    #     ret = {k: 0 for k in self.metric_keys}
    #     ret['n_samples'] = 0
    #     img_hr = sample['img_hr']
    #     img_lr = sample['img_lr']
    #     img_lr_up = sample['img_lr_up']
    #     sam_mask = sample['sam_mask']
    #
    #     img_sr, rrdb_out = self.model.sample(img_lr, img_lr_up, img_hr.shape, sam_mask=sam_mask)
    #
    #     for b in range(img_sr.shape[0]):
    #         s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])
    #         ret['psnr'] += s['psnr']
    #         ret['ssim'] += s['ssim']
    #         ret['lpips'] += s['lpips']
    #         ret['lr_psnr'] += s['lr_psnr']
    #         ret['n_samples'] += 1
    #     return img_sr, rrdb_out, ret
    
    def training_step(self, batch):
        img_hr = batch['img_hr']
        img_lr = batch['img_lr']
        img_lr_up = batch['img_lr_up']
        sam_mask = batch['sam_mask']
        losses, _, _ = self.model(img_hr, img_lr, img_lr_up, sam_mask=sam_mask)
        total_loss = sum(losses.values())
        return losses, total_loss
