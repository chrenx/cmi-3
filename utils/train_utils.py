from model.encoder_decoder.encoder_decoder import EncoderDecoder


class ModelLoader:
    def __init__(self, opt):
        self.model_name = ''.join(opt.model_name.split('_')[:-1]) # remove version str: unet_v1
        self.model = None
        self._load_model()
        
    def _load_model(self, opt):
        match self.model_name:
            case 'encoder_decoder':
                self.model = EncoderDecoder(
                    
                )
            case _:
                raise ValueError(f"Unknown model type in experiment name: {self.model_name}")


        
        assert self.model is not None, "Error when loading model"


#***************************************************************************************************

def cycle_dataloader(dl):
    while True:
        for data in dl:
            # print(data.shape)
            yield data

