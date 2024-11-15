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
        print("before gaussian: ")
        print(self.input[11, 19])
        features = self.input[:,:CUTOFF[f"model_{self.model_n}"]]
        demographic_data = self.input[:,CUTOFF[f"model_{self.model_n}"]:]

        augmented_features = features + np.random.normal(loc=self.opt.gauss_mean, 
                                                         scale=self.opt.gauss_std, 
                                                         size=features.shape)

        # Recombine demographic and processed features into one dataset
        augmented_data = np.column_stack([augmented_features, demographic_data])

        print("Augmented Data Shape:", augmented_data.shape)

        self.input = augmented_data
        print("after gaussian: ")
        print(self.input[11, 19])
                    
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return None
    
    

