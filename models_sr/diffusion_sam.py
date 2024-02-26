import torch
import torch.nn.functional as F
from thop import profile
from tqdm import tqdm

from utils_sr.hparams import hparams
from .diffusion import GaussianDiffusion, noise_like, extract
from .module_util import default


def get_flops(model, inputs):
    flops, params = profile(model, inputs=inputs)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


class GaussianDiffusion_sam(GaussianDiffusion):
    def __init__(self, denoise_fn, rrdb_net, timesteps=1000, loss_type='l1', sam_config=None):
        super().__init__(denoise_fn, rrdb_net, timesteps, loss_type)
        self.sam_config = sam_config
    
    def p_losses(self, x_start, t, cond, img_lr_up, noise=None, sam_mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        if self.sam_config['p_losses_sam']:
            _sam_mask = F.interpolate(sam_mask, noise.shape[2:], mode='bilinear')
            if self.sam_config.get('mask_coefficient', False):
                _sam_mask *= extract(self.mask_coefficient.to(_sam_mask.device), t, x_start.shape)
            noise += _sam_mask
        
        x_tp1_gt = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_t_gt = self.q_sample(x_start=x_start, t=t - 1, noise=noise)
        noise_pred = self.denoise_fn(x_tp1_gt, t, cond, img_lr_up, sam_mask=sam_mask)
        x_t_pred, x0_pred = self.p_sample(x_tp1_gt, t, cond, img_lr_up, noise_pred=noise_pred, sam_mask=sam_mask)
        
        if self.loss_type == 'l1':
            loss = (noise - noise_pred).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)
        elif self.loss_type == 'ssim':
            loss = (noise - noise_pred).abs().mean()
            loss = loss + (1 - self.ssim_loss(noise, noise_pred))
        else:
            raise NotImplementedError()
        return loss, x_tp1_gt, noise_pred, x_t_pred, x_t_gt, x0_pred
    
    @torch.no_grad()
    def p_sample(self, x, t, cond, img_lr_up, noise_pred=None, clip_denoised=True, repeat_noise=False, sam_mask=None):
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond, img_lr_up=img_lr_up, sam_mask=sam_mask)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
                x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised)
        
        noise = noise_like(x.shape, device, repeat_noise)
        
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0_pred
    
    @torch.no_grad()
    def sample(self, img_lr, img_lr_up, shape, sam_mask=None, save_intermediate=False):
        device = self.betas.device
        b = shape[0]
        
        if not hparams['res']:
            t = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)
            noise = None
            img = self.q_sample(img_lr_up, t, noise=noise)
        else:
            img = torch.randn(shape, device=device)
        
        if hparams['use_rrdb']:
            rrdb_out, cond = self.rrdb(img_lr, True)
        else:
            rrdb_out = img_lr_up
            cond = img_lr
        
        it = reversed(range(0, self.num_timesteps))
        
        if self.sample_tqdm:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)
        
        images = []
        for i in it:
            img, x_recon = self.p_sample(
                    img, torch.full((b,), i, device=device, dtype=torch.long), cond, img_lr_up, sam_mask=sam_mask)
            if save_intermediate:
                img_ = self.res2img(img, img_lr_up)
                x_recon_ = self.res2img(x_recon, img_lr_up)
                images.append((img_.cpu(), x_recon_.cpu()))
        img = self.res2img(img, img_lr_up)
        
        if save_intermediate:
            return img, rrdb_out, images
        else:
            return img, rrdb_out
