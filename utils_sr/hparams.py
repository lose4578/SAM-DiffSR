import argparse
import os

import yaml

global_print_hparams = True
hparams = {}


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def override_config(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v


def set_hparams(config='', exp_name='', hparams_str='', print_hparams=True, global_hparams=True):
    if config == '' and exp_name == '':
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--config', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--exp_name', type=str, default='', help='exp_name')
        parser.add_argument('--work_dir', type=str, default='', help='work dir')
        parser.add_argument('--gt_img_path', type=str, default='data/sr_diff/benchmark', help='gt_img_path')
        parser.add_argument('-hp', '--hparams', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--infer', action='store_true', help='infer')
        parser.add_argument('--benchmark', action='store_true', help='test benchmark')
        parser.add_argument('--benchmark_loop', action='store_true', help='loop test benchmark for all checkpoint')
        parser.add_argument('--benchmark_name_list', nargs='+',
                            default=['test_Set5', 'test_Set14', 'test_Urban100', 'test_Manga109', 'test_BSDS100'])
        parser.add_argument('--metric_list', nargs='+', default=['psnr-Y', 'ssim', 'fid'])
        parser.add_argument('--validate', action='store_true', help='validate')
        parser.add_argument('--val_steps', type=int, default=None, help='validate steps')
        parser.add_argument('--reset', action='store_true', help='reset hparams')
        parser.add_argument('--debug', action='store_true', help='debug')
        
        parser.add_argument('--img_dir', type=str, default='', help='infer input image dir')
        parser.add_argument('--save_dir', type=str, default='', help='infer output image dir')
        parser.add_argument('--ckpt_path', type=str, default='', help='infer ckpt path')
        
        
        args, unknown = parser.parse_known_args()
        print("| Unknow hparams: ", unknown)
    else:
        args = Args(config=config, exp_name=exp_name, hparams=hparams_str,
                    infer=False, validate=False, reset=False, debug=False)
    global hparams
    assert args.config != '' or args.exp_name != ''
    if args.config != '':
        assert os.path.exists(args.config)
    
    config_chains = []
    loaded_config = set()
    
    def load_config(config_fn):
        # deep first inheritance and avoid the second visit of one node
        if not os.path.exists(config_fn):
            return {}
        with open(config_fn) as f:
            hparams_ = yaml.safe_load(f)
        loaded_config.add(config_fn)
        if 'base_config' in hparams_:
            ret_hparams = {}
            if not isinstance(hparams_['base_config'], list):
                hparams_['base_config'] = [hparams_['base_config']]
            for c in hparams_['base_config']:
                if c.startswith('.'):
                    c = f'{os.path.dirname(config_fn)}/{c}'
                    c = os.path.normpath(c)
                if c not in loaded_config:
                    override_config(ret_hparams, load_config(c))
            override_config(ret_hparams, hparams_)
        else:
            ret_hparams = hparams_
        config_chains.append(config_fn)
        return ret_hparams
    
    saved_hparams = {}
    args_work_dir = ''
    if args.exp_name != '':
        args_work_dir = os.path.join(args.work_dir, 'checkpoints', args.exp_name)
        ckpt_config_path = f'{args_work_dir}/config.yaml'
        if os.path.exists(ckpt_config_path):
            with open(ckpt_config_path) as f:
                saved_hparams_ = yaml.safe_load(f)
                if saved_hparams_ is not None:
                    saved_hparams.update(saved_hparams_)
    hparams_ = {}
    if args.config != '':
        hparams_.update(load_config(args.config))
    if not args.reset:
        hparams_.update(saved_hparams)
    hparams_['work_dir'] = args_work_dir
    
    # Support config overriding in command line. Support list type config overriding.
    # Examples: --hparams="a=1,b.c=2,d=[1 1 1]"
    if args.hparams != "":
        for new_hparam in args.hparams.split(","):
            k, v = new_hparam.split("=")
            v = v.strip("\'\" ")
            config_node = hparams_
            for k_ in k.split(".")[:-1]:
                config_node = config_node[k_]
            k = k.split(".")[-1]
            if k not in config_node:
                config_node[k] = v
            
            elif v in ['True', 'False'] or type(config_node[k]) in [bool, list, dict]:
                if type(config_node[k]) == list:
                    v = v.replace(" ", ",")
                config_node[k] = eval(v)
            else:
                config_node[k] = type(config_node[k])(v)
    if args_work_dir != '' and (not os.path.exists(ckpt_config_path) or args.reset) and not args.infer:
        os.makedirs(hparams_['work_dir'], exist_ok=True)
        with open(ckpt_config_path, 'w') as f:
            yaml.safe_dump(hparams_, f)
    
    hparams_['infer'] = args.infer
    hparams_['debug'] = args.debug
    hparams_['validate'] = args.validate
    hparams_['exp_name'] = args.exp_name
    hparams_['val_steps'] = args.val_steps
    hparams_['benchmark'] = args.benchmark
    hparams_['benchmark_loop'] = args.benchmark_loop
    hparams_['benchmark_name_list'] = args.benchmark_name_list
    hparams_['gt_img_path'] = args.gt_img_path
    hparams_['metric_list'] = args.metric_list
    
    hparams_['img_dir'] = args.img_dir
    hparams_['save_dir'] = args.save_dir
    hparams_['ckpt_path'] = args.ckpt_path
    
    global global_print_hparams
    if global_hparams:
        hparams.clear()
        hparams.update(hparams_)
    if print_hparams and global_print_hparams and global_hparams:
        print('| Hparams chains: ', config_chains)
        print('| Hparams: ')
        for i, (k, v) in enumerate(sorted(hparams_.items())):
            print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
        print("")
        global_print_hparams = False
    return hparams_
