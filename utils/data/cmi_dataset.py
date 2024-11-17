import os

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.globals import CUTOFF


class CMIDataset(Dataset):
    '''
    CMI pertusis boost, vaccince response dataset.
    '''
    
    def __init__(self, opt):
        """Initialize CMIDataset.

        Args:
            opt.root_dpath: 'data/processed_data'
        """

        self.device = opt.device
        self.model_n = opt.model_n
        self.data_dir = opt.data_dir
        self.data_aug_gauss_noise = opt.data_aug_gauss_noise
        self.challenge_mode = opt.challenge_mode
        self.gauss_mean = opt.gauss_mean
        self.gauss_std = opt.gauss_std
        self.preload_gpu = opt.preload_gpu
        self.device = opt.device

        print(f"Initialize CMI dataset ---------------------------------------")

        if self.challenge_mode:
            self.input = np.load(os.path.join(self.data_dir, 
                                              f"model_{self.model_n}_challenge_input.npy"))
            if self.preload_gpu:
                self.input = torch.tensor(self.input).to(self.device)
        else:
            self.input = np.load(os.path.join(self.data_dir, f"model_{self.model_n}_input.npy"))
            self.gt = np.load(os.path.join(self.data_dir, f"model_{self.model_n}_target.npy"))
            self.gt_mask = np.load(os.path.join(self.data_dir, 
                                                    f"model_{self.model_n}_target_mask.npy"))
            if self.data_aug_gauss_noise:
                self._apply_gausion_noise()

            if self.preload_gpu:
                self.input = torch.tensor(self.input).to(self.device)

                #! do we actually need?
                self.gt = torch.tensor(self.gt).to(self.device)
                self.gt_mask = torch.tensor(self.gt_mask).to(self.device)

            print(f"Finish loading model {self.model_n} data -----------------------")
            print(f"    input      : {self.input.shape}")
            print(f"    gt     : {self.gt.shape}")
            print(f"    gt_mask: {self.gt_mask.shape}")
        
        opt.input_dim = self.input.shape[1]

        print()
        

    def _apply_gausion_noise(self):
        features = self.input[:,:CUTOFF[f"model_{self.model_n}"]]
        demographic_data = self.input[:,CUTOFF[f"model_{self.model_n}"]:]

        augmented_features = features + np.random.normal(loc=self.gauss_mean, 
                                                         scale=self.gauss_std, 
                                                         size=features.shape)
        augmented_data = np.column_stack([augmented_features, demographic_data])
        self.input = augmented_data
        print("Gaussian noised has been applied to input")
                    
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        sample = {
            "input": self.input[idx],
        }

        if self.opt.challenge_mode:
            sample["gt"] = self.gt[idx]
            sample["gt_mask"] = self.gt_mask[idx]

        return sample
    
    

