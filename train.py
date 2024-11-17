import glob, logging, os, time

import torch, wandb
import numpy as np
from torch.optim import Adam, AdamW, SGD
from torch.utils import data
from torch.utils.data import Subset

from utils.data.cmi_dataset import CMIDataset
from utils.setting import parse_opt, create_dirs_save_files, set_seed, set_redirect_printing
from utils.train_utils import ModelLoader, correlation_score, cycle_dataloader

os.environ['TZ'] = 'America/New_York'
time.tzset()

MYLOGGER = logging.getLogger()


class Trainer(object):
    def __init__(self, opt):

        super().__init__()
        
        self.device = opt.device
        self.use_wandb = not opt.disable_wandb
        self.weights_dir = opt.weights_dir
        self.warmup_steps = opt.lr_scheduler_warmup_steps
        self.disable_scheduler = opt.disable_scheduler
        self.penalty_cost = opt.penalty_cost
        self.train_n_steps = opt.train_n_steps

        self.eps = opt.eps

        self.train_n_batch, self.val_n_batch, self.test_n_batch = None, None, None
        self.train_dl, self.val_dl, self.test_dl = None, None, None
        self._prepare_dataloader(opt)

        #TODO-=-=-=-=-=-=-=-=------------=-===================================-=--=-=-=-=-=-=-=-=
        self.model = ModelLoader(opt).model.to(self.device)
        self.corr_loss_func = correlation_score
        self.mse_loss_func = nn.MSELoss()
        self.optimizer = self._get_optimizer(opt)
        self.scheduler_optim = self._get_lr_scheduler(opt) if not self.disable_scheduler else None

        if self.use_wandb:
            MYLOGGER.info("Initialize W&B")
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, 
                       name=opt.exp_name, dir=opt.save_dir)
            opt.wandb = wandb

        print("len train dataset:", len(self.train_ds))
        print("train_n_batch:", self.train_n_batch)
        print("len val dataset:", len(self.val_ds))
        print("val_n_batch:", self.val_n_batch)
        print("len test dataset:", len(self.test_ds))
        print("test_n_batch:", self.test_n_batch)
        print("lr: ", self.lr)

        print()

        exit(0)
        
    def _prepare_dataloader(self, opt):
        self.dataset = CMIDataset(opt)
        train_idx = np.load("data/processed_data/model_1_train_idx.npy")
        val_idx = np.load("data/processed_data/model_1_val_idx.npy")
        test_idx = np.load("data/processed_data/model_1_test_idx.npy")
        self.train_ds = Subset(self.dataset, train_idx)
        self.val_ds = Subset(self.dataset, val_idx)
        self.test_ds = Subset(self.dataset, test_idx)
    
        dl = data.DataLoader(self.train_ds, 
                            batch_size=opt.batch_size, 
                            shuffle=True, 
                            pin_memory=False, 
                            num_workers=0)
        self.train_n_batch = len(dl) 
        self.train_dl = cycle_dataloader(dl)

        dl = data.DataLoader(self.val_ds, 
                            batch_size=opt.batch_size, 
                            shuffle=False, 
                            pin_memory=False, 
                            num_workers=0)
        self.val_n_batch = len(dl)
        self.val_dl = cycle_dataloader(dl)

        dl = data.DataLoader(self.test_ds, 
                            batch_size=opt.batch_size, 
                            shuffle=False, 
                            pin_memory=False, 
                            num_workers=0)
        self.test_n_batch = len(dl)
        self.test_dl = cycle_dataloader(dl)

        



    def _get_lr_scheduler(opt):
        match opt.lr_scheduler:
            case 'ReduceLROnPlateau':
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer=self.optimizer,
                            factor=opt.lr_scheduler_factor,
                            patience=opt.lr_scheduler_patience)
            case 'MultiStepLR':
                return torch.optim.lr_scheduler.MultiStepLR(
                            optimizer=self.optimizer,
                            milestones=[10,20,30,40,50,60,70,80,90])
            case 'MultiplicativeLR':
                lmbda = lambda epoch: 0.65 ** epoch
                return torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lmbda)
            case 'CosineAnnealingLR':
                return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=10,eta_min=0)
            case _ :
                raise "Pick a learning rate scheduler."

    def _get_optimizer(self, opt):
        match opt.optimizer:
            case 'adam':
                return Adam(self.model.parameters(), lr=opt.lr, eps=opt.eps, 
                            weight_decay=opt.weight_decay)
            case _:
                raise "Give a proper name of optimizer"

    def _save_model(self, step, base, best=False):
        data = {
            'step': step,
            'model': self.model.state_dict(),
        }
        # delete previous best* or model*
        if best: 
            search_pattern = os.path.join(self.weights_dir, f"best_model_{base}*")
        else:
            search_pattern = os.path.join(self.weights_dir, "model*")
        files_to_delete = glob.glob(search_pattern)

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                MYLOGGER.error(f"Error deleting file {file_path}: {e}")
            
        filename = f"best_model_{base}_{step}.pt" if best else f"model_{base}_{step}.pt"
        torch.save(data, os.path.join(self.weights_dir, filename))      
        
    def _loss_func(self, pred, gt, gt_mask=None, train_mode=False):
        """

        Args:
            pred: (B, 5) value
            gt: (B, 5) value
        """

        penalty_cost = self.penalty_cost if train_mode else 0
        loss_corr = self.corr_loss_func(pred * gt_mask, gt * gt_mask)
        loss_mse = self.mse_loss_func(pred * gt_mask, gt * gt_mask)
        
        mask = (gt[:,:,2] != 1).float() # (B, window)
        mask = mask.unsqueeze(-1) # (B, window, 1)

        #TODO: check mask size !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # Additional cost for misclassifying the minority class
        minority_mask = (gt[:,:,1] == 1).float() # (B, window)
        minority_mask = minority_mask.unsqueeze(-1) # (B, window, 1)
        loss = loss * (mask + penalty_cost * minority_mask)
   
        return loss.sum() / mask.sum()

    def _evaluation_metrics(self, output, gt):
        """Generate precision, recall, and f1 score.

        Args:
            output: (B,5) 
            gt (inference): (B,5)
        """
        # Convert the model output probabilities to class predictions
        pred = torch.round(output)  # (B, window, 1)

        # Extract the first two classes from the ground truth
        real = torch.argmax(gt[:, :, :2], dim=-1, keepdim=True)  # (B, window, 1)

        # Create a mask to ignore the positions where the ground truth class is 2
        mask = (gt[:, :, 2] != 1).unsqueeze(-1)  # (B, window, 1)

        # Apply the mask to the predictions and ground truth
        pred = (pred * mask.float()).squeeze() # (B, window)
        real = (real * mask.float()).squeeze() # (B, window)
        

        # Calculate true positives, false positives, and false negatives
        tp = ((pred == 1) & (real == 1)).float().sum()
        fp = ((pred == 1) & (real == 0)).float().sum()
        fn = ((pred == 0) & (real == 1)).float().sum()

        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return precision, recall, f1

    def _eval_test_data(self, step, base):
        avg_test_loss = 0.0      
        self.model.eval()
        with torch.no_grad():
            for _ in tqdm(range(self.test_n_batch), desc=f"test data"):
                test_data = next(self.test_dl) 

                test_gt_mask = test_data["gt_mask"]
                test_gt = test_data["gt"] # (B,5)
                test_input = test_data["input"]    
                
                test_pred = self.model(test_input) # (B, window, 1)
                
                #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                prec, recall, f1 = self._evaluation_metrics(test_pred, test_gt.to(self.device))
                test_loss = self._loss_func(test_pred, test_gt.to(self.device), 
                                            test_gt_mask.to(self.device))
                
                avg_test_loss += test_loss
                
            avg_test_loss /= self.test_n_batch

            avg_test_loss = avg_test_loss.item()
            
            MYLOGGER.info(f"avg_test_loss: {avg_test_loss}")
            
            if self.use_wandb:
                wandb.run.summary[f'best_test_loss_from_val_{base} / step'] = [avg_test_loss, step]

    def _do_challenge(self, step, base):
        #TODO

        pass

    def train(self):
        best_val_loss = float('inf')

        for step_idx in tqdm(range(0, self.train_n_steps), desc="Train"):
            self.model.train()
            
            #* training part -----------------------------------------------------------------------
            train_data = next(self.train_dl)
            
            train_gt_mask = train_data["gt_mask"] # (B,5)
            train_gt = train_data["gt"] # (B,5)      
            train_input = train_data["input"] # (B,445/419) 

            train_pred = self.model(train_input) # (B,5)
            
            train_loss = self._loss_func(train_pred, train_gt.to(self.device), 
                                         train_gt_mask.to(self.device), train_mode=True)

            train_loss.backward()
            
            # check gradients
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).\
                                        to(self.device) for p in parameters]), 2.0)

            if torch.isnan(total_norm):
                MYLOGGER.warning('NaN gradients. Skipping to next data...')
                torch.cuda.empty_cache()
                continue
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.use_wandb:
                log_dict = {
                    "Train/loss": train_loss.item(),
                }
                wandb.log(log_dict, step=step_idx+1)
            
            if not self.preload_gpu:
                torch.cuda.empty_cache()
                
    
            #* validation part ---------------------------------------------------------------------
            cur_epoch = (step_idx + 1) // self.train_n_batch
            if not self.use_wandb or (step_idx + 1) % self.train_n_batch == 0: #* an epoch
            # if True:
                avg_val_loss = 0.0
                
                self.model.eval()
                with torch.no_grad():
                    for _ in tqdm(range(self.val_n_batch), desc=f"Validation at epoch {cur_epoch}"):
                        
                        val_data = next(self.val_dl)

                        val_gt_mask = val_data["gt_mask"] # (B,5)
                        val_gt = val_data["gt"] # (B,5)
                        val_input = val_data["input"]
                        val_pred = self.model(val_input) # (B, window, 1)
                        
                        prec, recall, f1 = self._evaluation_metrics(val_pred, 
                                                                    val_gt.to(self.device))
                        val_loss = self._loss_func(val_pred, val_gt.to(self.device), 
                                                   val_gt_mask.to(self.device))
                        
                        avg_val_loss += val_loss
                
                    avg_val_loss /= self.val_n_batch
    
                    
                    if self.use_wandb:
                        log_dict = {
                            "Val/avg_val_loss": avg_val_loss.item(),
                        }
                        wandb.log(log_dict, step=step_idx+1)
        
                    MYLOGGER.info(f"avg_val_loss: {avg_val_loss.item():4f}")

                # Log learning rate
                if self.use_wandb and not self.disable_scheduler:
                    wandb.log({'learning_rate': self.scheduler_optim.get_last_lr()[0]}, 
                              step=step_idx+1)
                
                if self.use_wandb and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if self.use_wandb:
                        wandb.run.summary['best_val_loss / step'] = [avg_val_loss.item(), 
                                                                     step_idx+1]
                    self._save_model(step_idx + 1, base='loss', best=True)
                    self._eval_test_data(step=step_idx + 1, base='loss')
                    
                self._save_model(step_idx + 1, base='regular', best=False)
            
                #* learning rate scheduler ---------------------------------------------------------
                if cur_epoch > self.warmup_steps and not self.disable_scheduler:    
                    self.scheduler_optim.step(avg_val_loss)
                    
                if not self.preload_gpu:
                    torch.cuda.empty_cache()

        if self.use_wandb:
            wandb.run.finish()


def main():
    assert torch.cuda.is_available(), "**** No available GPUs."
    opt = parse_opt()
    create_dirs_save_files(opt)
    set_seed(opt.seed)
    set_redirect_printing(opt)

    trainer = Trainer(opt)
    trainer.train()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
