import importlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

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
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils_sr.hparams import hparams, set_hparams
from utils_sr.utils import plot_img, move_to_cuda, load_checkpoint, save_checkpoint, tensors_to_scalars, Measure, \
    get_all_ckpts
from tools.caculate_iqa import eval_img_IQA


class Trainer:
    def __init__(self):
        self.logger = self.build_tensorboard(save_dir=hparams['work_dir'], name='tb_logs')
        self.measure = Measure()
        self.dataset_cls = None
        self.metric_keys = ['psnr', 'ssim', 'lpips', 'lr_psnr']
        self.metric_2_keys = ['psnr-Y', 'ssim', 'fid']
        self.work_dir = hparams['work_dir']
        self.first_val = True
        
        self.val_steps = hparams['val_steps']
    
    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir, **kwargs)
    
    def build_train_dataloader(self):
        dataset = self.dataset_cls('train')
        return torch.utils.data.DataLoader(
                dataset, batch_size=hparams['batch_size'], shuffle=True,
                pin_memory=False, num_workers=hparams['num_workers'])
    
    def build_val_dataloader(self):
        return torch.utils.data.DataLoader(
                self.dataset_cls('valid'), batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)
    
    def build_test_dataloader(self):
        return torch.utils.data.DataLoader(
                self.dataset_cls('test'), batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)
    
    def build_model(self):
        raise NotImplementedError
    
    def sample_and_test(self, sample):
        raise NotImplementedError
    
    def build_optimizer(self, model):
        raise NotImplementedError
    
    def build_scheduler(self, optimizer):
        raise NotImplementedError
    
    def training_step(self, batch):
        raise NotImplementedError
    
    def train(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        self.global_step = training_step = load_checkpoint(model, optimizer, hparams['work_dir'], steps=self.val_steps)
        self.scheduler = scheduler = self.build_scheduler(optimizer)
        scheduler.step(training_step)
        dataloader = self.build_train_dataloader()
        
        train_pbar = tqdm(dataloader, initial=training_step, total=float('inf'),
                          dynamic_ncols=True, unit='step')
        while self.global_step < hparams['max_updates']:
            for batch in train_pbar:
                if training_step % hparams['val_check_interval'] == 0:
                    with torch.no_grad():
                        model.eval()
                        self.validate(training_step)
                    save_checkpoint(model, optimizer, self.work_dir, training_step, hparams['num_ckpt_keep'])
                model.train()
                batch = move_to_cuda(batch)
                losses, total_loss = self.training_step(batch)
                optimizer.zero_grad()
                
                total_loss.backward()
                optimizer.step()
                training_step += 1
                scheduler.step(training_step)
                self.global_step = training_step
                if training_step % 100 == 0:
                    self.log_metrics({f'tr/{k}': v for k, v in losses.items()}, training_step)
                train_pbar.set_postfix(**tensors_to_scalars(losses))
    
    def validate(self, training_step):
        val_dataloader = self.build_val_dataloader()
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        metrics = {}
        for batch_idx, batch in pbar:
            # 每次运行的第一次validation只跑一小部分数据，来验证代码能否跑通
            if self.first_val and batch_idx > hparams['num_sanity_val_steps'] - 1:
                break
            batch = move_to_cuda(batch)
            img, rrdb_out, ret = self.sample_and_test(batch)
            img_hr = batch['img_hr']
            img_lr = batch['img_lr']
            img_lr_up = batch['img_lr_up']
            if img is not None:
                self.logger.add_image(f'Pred_{batch_idx}', plot_img(img[0]), self.global_step)
                if hparams.get('aux_l1_loss'):
                    self.logger.add_image(f'rrdb_out_{batch_idx}', plot_img(rrdb_out[0]), self.global_step)
                if self.global_step <= hparams['val_check_interval']:
                    self.logger.add_image(f'HR_{batch_idx}', plot_img(img_hr[0]), self.global_step)
                    self.logger.add_image(f'LR_{batch_idx}', plot_img(img_lr[0]), self.global_step)
                    self.logger.add_image(f'BL_{batch_idx}', plot_img(img_lr_up[0]), self.global_step)
            metrics = {}
            metrics.update({k: np.mean(ret[k]) for k in self.metric_keys})
            pbar.set_postfix(**tensors_to_scalars(metrics))
        if hparams['infer']:
            print('Val results:', metrics)
        else:
            if not self.first_val:
                self.log_metrics({f'val/{k}': v for k, v in metrics.items()}, training_step)
                print('Val results:', metrics)
            else:
                print('Sanity val results:', metrics)
        self.first_val = False
    
    def build_test_my_dataloader(self, data_name):
        return torch.utils.data.DataLoader(
                self.dataset_cls(data_name), batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)
    
    def benchmark(self, benchmark_name_list, metric_list):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        training_step = load_checkpoint(model, optimizer, hparams['work_dir'], hparams['val_steps'])
        self.global_step = training_step
        
        optimizer = None
        
        for data_name in benchmark_name_list:
            test_dataloader = self.build_test_my_dataloader(data_name)
            
            self.results = {k: 0 for k in self.metric_keys}
            self.n_samples = 0
            self.gen_dir = f"{hparams['work_dir']}/results_{self.global_step}_{hparams['gen_dir_name']}/benchmark/{data_name}"
            if hparams['test_save_png']:
                subprocess.check_call(f'rm -rf {self.gen_dir}', shell=True)
                os.makedirs(f'{self.gen_dir}/outputs', exist_ok=True)
                os.makedirs(f'{self.gen_dir}/SR', exist_ok=True)
            
            self.model.sample_tqdm = False
            torch.backends.cudnn.benchmark = False
            if hparams['test_save_png']:
                if hasattr(self.model.denoise_fn, 'make_generation_fast_'):
                    self.model.denoise_fn.make_generation_fast_()
                os.makedirs(f'{self.gen_dir}/HR', exist_ok=True)
            
            result_dict = {}
            
            with torch.no_grad():
                model.eval()
                pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
                for batch_idx, batch in pbar:
                    move_to_cuda(batch)
                    gen_dir = self.gen_dir
                    item_names = batch['item_name']
                    img_hr = batch['img_hr']
                    img_lr = batch['img_lr']
                    img_lr_up = batch['img_lr_up']
                    
                    res = self.sample_and_test(batch)
                    if len(res) == 3:
                        img_sr, rrdb_out, ret = res
                    else:
                        img_sr, ret = res
                        rrdb_out = img_sr
                    
                    img_lr_up = batch.get('img_lr_up', img_lr_up)
                    if img_sr is not None:
                        metrics = list(self.metric_keys)
                        result_dict[batch['item_name'][0]] = {}
                        for k in metrics:
                            self.results[k] += ret[k]
                            result_dict[batch['item_name'][0]][k] = ret[k]
                        self.n_samples += ret['n_samples']
                        
                        print({k: round(self.results[k] / self.n_samples, 3) for k in self.results}, 'total:',
                              self.n_samples)
                        
                        if hparams['test_save_png'] and img_sr is not None:
                            img_sr = self.tensor2img(img_sr)
                            img_hr = self.tensor2img(img_hr)
                            img_lr = self.tensor2img(img_lr)
                            img_lr_up = self.tensor2img(img_lr_up)
                            rrdb_out = self.tensor2img(rrdb_out)
                            for item_name, hr_p, hr_g, lr, lr_up, rrdb_o in zip(
                                    item_names, img_sr, img_hr, img_lr, img_lr_up, rrdb_out):
                                item_name = os.path.splitext(item_name)[0]
                                hr_p = Image.fromarray(hr_p)
                                hr_g = Image.fromarray(hr_g)
                                hr_p.save(f"{gen_dir}/SR/{item_name}.png")
                                hr_g.save(f"{gen_dir}/HR/{item_name}.png")
            
            exp_name = hparams['work_dir'].split('/')[-1]
            sr_img_dir = f"{gen_dir}/SR/"
            gt_img_dir = f"{gen_dir}/HR/"
            excel_path = f"{hparams['work_dir']}/IQA-val-benchmark-{exp_name}.xlsx"
            epoch = training_step
            eval_img_IQA(gt_img_dir, sr_img_dir, excel_path, metric_list, epoch, data_name)
            
            os.makedirs(f'{self.gen_dir}', exist_ok=True)
            eval_json_path = os.path.join(self.gen_dir, 'eval.json')
            avg_result = {k: round(self.results[k] / self.n_samples, 4) for k in self.results}
            with open(eval_json_path, 'w+') as file:
                json.dump(avg_result, file, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
                json.dump(result_dict, file, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
    
    def benchmark_loop(self, benchmark_name_list, metric_list, gt_path):
        # infer and evaluation all save checkpoint
        
        model = self.build_model()
        
        def get_checkpoint(model, checkpoint):
            stat_dict = checkpoint['state_dict']['model']
            
            new_state_dict = OrderedDict()
            for k, v in stat_dict.items():
                if k[:7] == 'module.':
                    k = k[7:]  # 去掉 `module.`
                new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
            model.cuda()
            training_step = checkpoint['global_step']
            del checkpoint
            torch.cuda.empty_cache()
            
            return training_step
        
        ckpt_paths = get_all_ckpts(hparams['work_dir'])
        for ckpt_path in ckpt_paths:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            training_step = get_checkpoint(model, checkpoint)
            
            self.global_step = training_step
            
            for data_name in benchmark_name_list:
                test_dataloader = self.build_test_my_dataloader(data_name)
                
                self.results = {k: 0 for k in self.metric_keys + self.metric_2_keys}
                self.n_samples = 0
                self.gen_dir = f"{hparams['work_dir']}/results_{training_step}_{hparams['gen_dir_name']}/benchmark/{data_name}"
                
                os.makedirs(f'{self.gen_dir}/outputs', exist_ok=True)
                os.makedirs(f'{self.gen_dir}/SR', exist_ok=True)
                
                self.model.sample_tqdm = False
                torch.backends.cudnn.benchmark = False
                
                with torch.no_grad():
                    model.eval()
                    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
                    for batch_idx, batch in pbar:
                        move_to_cuda(batch)
                        gen_dir = self.gen_dir
                        item_names = batch['item_name']
                        
                        res = self.sample_and_test(batch)
                        if len(res) == 3:
                            img_sr, rrdb_out, ret = res
                        else:
                            img_sr, ret = res
                            rrdb_out = img_sr
                        
                        img_sr = self.tensor2img(img_sr)
                        
                        for item_name, hr_p in zip(item_names, img_sr):
                            item_name = os.path.splitext(item_name)[0]
                            hr_p = Image.fromarray(hr_p)
                            hr_p.save(f"{gen_dir}/SR/{item_name}.png")
                
                exp_name = hparams['work_dir'].split('/')[-1]
                sr_img_dir = f"{gen_dir}/SR/"
                gt_img_dir = f"{gt_path}/{data_name}/HR"
                excel_path = f"{hparams['work_dir']}/IQA-val-benchmark_loop-{exp_name}.xlsx"
                epoch = training_step
                eval_img_IQA(gt_img_dir, sr_img_dir, excel_path, metric_list, epoch, data_name)
    
    # utils_sr
    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)
    
    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            
            if type(v) is dict:
                v = self.metrics_to_scalars(v)
            
            new_metrics[k] = v
        
        return new_metrics
    
    @staticmethod
    def tensor2img(img):
        img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img


if __name__ == '__main__':
    set_hparams()
    
    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    cls_name = hparams["trainer_cls"].split(".")[-1]
    trainer = getattr(importlib.import_module(pkg), cls_name)()
    if hparams['benchmark_loop']:
        trainer.benchmark_loop(hparams['benchmark_name_list'], hparams['metric_list'], hparams['gt_img_path'])
    elif hparams['benchmark']:
        trainer.benchmark(hparams['benchmark_name_list'], hparams['metric_list'])
    else:
        trainer.train()
