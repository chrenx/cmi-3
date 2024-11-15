import argparse, logging, os, random, shutil, sys, yaml
from datetime import datetime

import torch
import numpy as np


class DotDict(dict):
    """Dictionary with dot notation access to nested keys."""
    def __init__(self, data=None):
        super().__init__()
        data = data or {}
        for key, value in data.items():
            # If the value is a dictionary, convert it to a DotDict recursively
            self[key] = DotDict(value) if isinstance(value, dict) else value

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = DotDict(value) if isinstance(value, dict) else value

    def __delattr__(self, attr):
        del self[attr]

#***************************************************************************************************
def create_dirs_save_files(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.weights_dir, exist_ok=True)
    os.makedirs(opt.codes_dir, exist_ok=True)
    with open(os.path.join(opt.codes_dir, "updated_config.yaml"), "w") as file:
        yaml.dump(opt, file, default_flow_style=False, sort_keys=False)
    # Save some important code
    source_files = [f'train.sh',
                    f'train.py']
    for file_dir in source_files:
        shutil.copy2(file_dir, opt.codes_dir)

    def ignore_patterns(place_holder, names):
        # Ignore any directories named '__pycache__'
        return ['__pycache__'] if '__pycache__' in names else []
    
    source_dirs = ['utils/', 'model/']
    for source_dir in source_dirs:
        source_item = os.path.join(source_dir)
        destination_item = os.path.join(opt.codes_dir, source_item)
        shutil.copytree(source_item, destination_item, dirs_exist_ok=True, ignore=ignore_patterns)

def get_cur_time():
    # output: e.g. 2024_11_01_13_14_01
    cur_time = datetime.now()
    cur_time = '{:%Y_%m_%d_%H_%M_%S}_{:02.0f}'.\
                format(cur_time, cur_time.microsecond / 10000.0)
    return cur_time

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # project information ==========================================================================
    parser.add_argument('--exp_name', type=str, help='wandb project name')
    parser.add_argument('--description', type=str, help='important notes')
    parser.add_argument('--save_dir', type=str, 
                             help='save important files, weights, etc., will be intialized if null')
    parser.add_argument('--wandb_pj_name', type=str, help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, help='wandb account')
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--seed', type=int, help='initializing seed')
    parser.add_argument('--debug', action='store_true')
    
    # data =========================================================================================
    parser.add_argument('--data_dir', type=str, help='data directory')

    # GPU ==========================================================================================
    parser.add_argument('--cuda_id', type=str, help='assign gpu')
    
    # training =====================================================================================
    parser.add_argument('--save_best_model', action='store_true', 
                                             help='save best model during training')
    parser.add_argument('--batch_size', type=int, )
    parser.add_argument('--lr', type=float, help='learning rate')

    # ==============================================================================================

    args = vars(parser.parse_args())
    cli_args = [k for k in args.keys() if args[k] is not None and args[k] is not False]
    
    with open('utils/config.yaml', 'r') as f:
        opt = yaml.safe_load(f)
        opt = DotDict(opt)
        for arg in cli_args: # update arguments if passed from command line; otherwise, use default
            opt[arg] = args[arg]

    cur_time = get_cur_time()
    opt.cur_time = cur_time
    if opt.save_dir is None:
        opt.save_dir = os.path.join('runs/train', opt.exp_name, cur_time)
    opt.weights_dir = os.path.join(opt.save_dir, 'weights')
    opt.codes_dir = os.path.join(opt.save_dir, 'codes')
    opt.device_info = torch.cuda.get_device_name(int(opt.cuda_id)) 
    opt.device = f"cuda:{opt.cuda_id}"

    return opt

def set_redirect_printing(opt):
    training_info_log_path = os.path.join(opt.save_dir, "training_info.log")
    
    if not opt.disable_wandb:
        sys.stdout = open(training_info_log_path, "w")
        
        logging.basicConfig(filename=os.path.join(training_info_log_path),
                    filemode='a',
                    format='%(asctime)s.%(msecs)02d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d-%H:%M:%S',
                    level=os.environ.get("LOGLEVEL", "INFO"))
    else:
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def set_seed(seed=None):
    assert seed is not None, "no seed is given"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

