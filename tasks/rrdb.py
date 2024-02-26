import torch
import torch.nn.functional as F

from models_sr.diffsr_modules import RRDBNet
from tasks.srdiff_df2k import Df2kDataSet
from tasks.trainer import Trainer
from utils_sr.hparams import hparams
from utils_sr.sr_utils import PerceptualLoss


class RRDBTask(Trainer):
    def __init__(self):
        super().__init__()
        if 'rrdb_loss' in hparams and hparams['rrdb_loss']['percep_loss']:
            self.percep_loss_fn = PerceptualLoss()
            self.percep_loss_weight = hparams['rrdb_loss']['percep_loss_weight']
        else:
            self.percep_loss_fn = None
            self.percep_loss_weight = 0
    
    def build_model(self):
        hidden_size = hparams['hidden_size']
        self.model = RRDBNet(3, 3, hidden_size, hparams['num_block'], hidden_size // 2)
        return self.model
    
    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    
    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 200000, 0.5)
    
    def training_step(self, sample):
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        p = self.model(img_lr)
        total_loss = 0
        loss = F.l1_loss(p, img_hr, reduction='mean')
        total_loss += loss
        
        if self.percep_loss_fn:
            loss_percep = self.percep_loss_fn(img_hr, p) * self.percep_loss_weight
            total_loss += loss_percep
            return {'l': loss, 'loss_percep': loss_percep, 'total_loss': total_loss,
                    'lr': self.scheduler.get_last_lr()[0]}, total_loss
        else:
            return {'l': loss, 'lr': self.scheduler.get_last_lr()[0]}, total_loss
    
    def sample_and_test(self, sample):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0
        img_hr = sample['img_hr']
        img_lr = sample['img_lr']
        img_sr = self.model(img_lr)
        img_sr = img_sr.clamp(-1, 1)
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])
            ret['psnr'] += s['psnr']
            ret['ssim'] += s['ssim']
            ret['lpips'] += s['lpips']
            ret['lr_psnr'] += s['lr_psnr']
            ret['n_samples'] += 1
        return img_sr, img_sr, ret


class RRDBDf2kTask(RRDBTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = Df2kDataSet
