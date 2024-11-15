import joblib, math, os, random

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

# from utils.globals import FEATS_1_filename, FEATS_2_filename


class CMIDataset(Dataset):
    '''
    CMI pertusis boost, vaccince response dataset.
    '''
    
    def __init__(self, opt, mode='train'):
        """Initialize CMIDataset.

        Args:
            opt.root_dpath: 'data/processed_data'
        """

        print(f"Initialize CMI dataset: {mode} mode --------")
        
        self.mode = mode
        self.device = opt.device
        self.opt = opt
        model_type = opt.model_type

        self.input = np.load(os.path.join(opt.data_dir, f"model_{model_type}_input.npy"))
        self.target = np.load(os.path.join(opt.data_dir, f"model_{model_type}_target.npy"))
        self.target_mask = np.load(os.path.join(opt.data_dir, 
                                                f"model_{model_type}_target_mask.npy"))
        
        print(f"Finish loading model {model_type} data ====================")
        print(f"    input      : {self.input.shape}")
        print(f"    target     : {self.target.shape}")
        print(f"    target_mask: {self.target_mask.shape}")
        print(f"    len      : {len(self.input)}")
        print()
        exit(0)
        
                    
    def __len__(self):
        return len(self.input_value)

    def __getitem__(self, idx):
        return self.window_data_dict[idx]
        # return self.window_data_dict[idx]['model_input']
    
    

