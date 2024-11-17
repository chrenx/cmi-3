import importlib

import torch

class ModelLoader:
    def __init__(self, opt):
        self.model = None
        self._load_model(opt)
        
    def _load_model(self, opt):
        model_name = ''.join(opt.model_name.split('_')[:-1]) # remove version str: unet_v1
        match model_name:
            case 'encoder_decoder':
                class_name = 'EncoderDecoder'
            case 'unet':
                class_name = 'UNet'
            case _:
                raise ValueError(f"Unknown model type in experiment name: {exp_name}")

        # Dynamically import the module and class
        module = importlib.import_module(f'models.{exp_name}.{opt.model_name}')
        model_class = getattr(module, class_name)

        self.model = model_class(opt)

        
        
        assert self.model is not None, "Error when loading model"


#***************************************************************************************************

def cycle_dataloader(dl):
    while True:
        for data in dl:
            # print(data.shape)
            yield data

def correlation_score(y_pred, y_true):
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:, None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:, None]
    cov_tp = torch.sum(y_true_centered * y_pred_centered, dim=1) / (y_true.shape[1] - 1)
    var_t = torch.sum(y_true_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    var_p = torch.sum(y_pred_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    score = cov_tp / torch.sqrt(var_t * var_p)
    return -torch.mean(score)


