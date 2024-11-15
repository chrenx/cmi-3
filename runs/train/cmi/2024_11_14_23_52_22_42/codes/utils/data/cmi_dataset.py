import os

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.globals import CUTOFF


class CMIDataset(Dataset):
    '''
    CMI pertusis boost, vaccince response dataset.
    '''
    
    def __init__(self, opt, mode='train'):
        """Initialize CMIDataset.

        Args:
            opt.root_dpath: 'data/processed_data'
        """

        print(f"Initialize CMI dataset ---------------------------------------")
        
        self.mode = mode
        self.device = opt.device
        self.opt = opt
        self.model_n = opt.model_n

        if self.opt.challenge_mode:
            self.input = np.load(os.path.join(opt.data_dir, 
                                              f"model_{self.model_n}_challenge_input.npy"))
        else:
            self.input = np.load(os.path.join(opt.data_dir, f"model_{self.model_n}_input.npy"))
            self.target = np.load(os.path.join(opt.data_dir, f"model_{self.model_n}_target.npy"))
            self.target_mask = np.load(os.path.join(opt.data_dir, 
                                                    f"model_{self.model_n}_target_mask.npy"))
            
            print(f"Finish loading model {self.model_n} data -----------------------")
            print(f"    input      : {self.input.shape}")
            print(f"    target     : {self.target.shape}")
            print(f"    target_mask: {self.target_mask.shape}")

            if self.opt.apply_gausion_noise:
                self._apply_gausion_noise()

        print()
        exit(0)
        

    def _apply_gausion_noise(self):
        features = self.input[:,:CUTOFF[f"model_{self.model_n}"]]
        demographic_data = self.input[:,CUTOFF[f"model_{self.model_n}"]:]

        augmented_features = features + np.random.normal(loc=self.opt.gauss_mean, 
                                                         scale=self.opt.gauss_std, 
                                                         size=features.shape)
        augmented_data = np.column_stack([augmented_features, demographic_data])
        self.input = augmented_data
                    
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        sample = {
            "input": self.input[idx],
        }

        if self.opt.challenge_mode:
            sample["gt"] = self.target[idx]
            sample["gt_mask"] = self.target_mask[idx]

        return sample
    
    

