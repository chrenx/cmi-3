import logging, os, time

import torch, wandb
import numpy as np
from torch.utils.data import Subset

from utils.data.cmi_dataset import CMIDataset
from utils.setting import parse_opt, create_dirs_save_files, set_seed, set_redirect_printing
from utils.train_utils import cycle_dataloader

os.environ['TZ'] = 'America/New_York'
time.tzset()

MYLOGGER = logging.getLogger()


class Trainer(object):
    def __init__(self, opt):
        super().__init__()
        
        # self.model = ModelLoader(opt.exp_name, opt).model
        # self.model.to(opt.device)
        
        self.use_wandb = not opt.disable_wandb
        self.weights_dir = opt.weights_dir
        # self.warmup_steps = opt.lr_scheduler_warmup_steps
        self.penalty_cost = opt.penalty_cost
        
        if self.use_wandb:
            MYLOGGER.info("Initialize W&B")
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, 
                       name=opt.exp_name, dir=opt.save_dir)
            opt.wandb = wandb
        
        self._prepare_dataloader(opt) 

        # self.optimizer = None
        # self.cur_step = 0
        # self.train_num_steps = opt.train_num_steps
        # self.device = opt.device
        # self.bce_loss = torch.nn.BCELoss(reduction='none')
        # self.max_grad_norm = opt.max_grad_norm
        # self.preload_gpu = opt.preload_gpu
        # self.grad_accum_step = opt.grad_accum_step
        # self.disable_scheduler = opt.disable_scheduler
        # self.opt = opt
        

        # cond = {
        #     'learning_rate': opt.learning_rate,
        #     'adam_betas': opt.adam_betas,
        #     'adam_eps': opt.adam_eps,
        #     'weight_decay': opt.weight_decay,
        #     'sgd_momentum': opt.sgd_momentum,
        #     'sgd_enable_nesterov': opt.sgd_enable_nesterov,
        # }

        # self.optimizer = get_optimizer(opt.optimizer, self.model.parameters(), cond)
        
        # cond = {
        #     'lr_scheduler_factor': opt.lr_scheduler_factor, 
        #     'lr_scheduler_patience': opt.lr_scheduler_patience
        # }
        # self.scheduler_optim = get_lr_scheduler(opt.lr_scheduler, self.optimizer, cond)
        
        # opt.model_num_params = count_model_parameters(self.model)
        
    def _prepare_dataloader(self, opt):

        self.dataset = CMIDataset(opt)
        train_idx = np.load("data/processed_data/model_1_train_idx.npy")
        val_idx = np.load("data/processed_data/model_1_val_idx.npy")
        test_idx = np.load("data/processed_data/model_1_test_idx.npy")
        train_dataset = Subset(self.dataset, train_idx)
        val_dataset = Subset(self.dataset, val_idx)
        test_dataset = Subset(self.dataset, test_idx)
        print("     len train dataset:", len(train_dataset))
        print("     len val dataset:", len(val_dataset))
        print("     len test dataset:", len(test_dataset))
        exit(0)
        
        # self.val_ds = CMIDataset(opt, mode='val')
        
        # self.test_ds = CMIDataset(opt, mode='test')
        
        # dl = data.DataLoader(self.train_ds, 
        #                     batch_size=opt.batch_size, 
        #                     shuffle=True, 
        #                     pin_memory=False, 
        #                     num_workers=0)
        # self.train_n_batch = len(dl) 
        # self.train_dl = cycle_dataloader(dl)
        # opt.train_n_batch = self.train_n_batch


        
        # dl = data.DataLoader(self.val_ds, 
        #                     batch_size=opt.batch_size, 
        #                     shuffle=False, 
        #                     pin_memory=False, 
        #                     num_workers=0)
        # self.val_n_batch = len(dl)
        # self.val_dl = cycle_dataloader(dl)
        # opt.val_n_batch = self.val_n_batch
        
        # dl = data.DataLoader(self.test_ds, 
        #                     batch_size=opt.batch_size, 
        #                     shuffle=False, 
        #                     pin_memory=False, 
        #                     num_workers=0)
        # self.test_n_batch = len(dl)
        # self.test_dl = cycle_dataloader(dl)
        # opt.test_n_batch = self.test_n_batch
        
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
        
    def _loss_func(self, pred, gt, train_mode=False):
        """Compute the Binary Cross-Entropy loss for each class and sum over the class dimension

        Args:
            pred: (B, window, 1) prob
            gt: (B, window, 3) one hot
        """
        penalty_cost = self.penalty_cost if train_mode else 0
        max_indices = torch.argmax(gt, dim=2, keepdim=True) # (B, window, 1)
        tmp_gt_mask = (max_indices != 2).float() # (B, window, 1)
        tmp_gt = max_indices * tmp_gt_mask # (B, window, 1)
        
        loss = self.bce_loss(pred, tmp_gt) # (B, window, 1)

        mask = (gt[:,:,2] != 1).float() # (B, window)
        mask = mask.unsqueeze(-1) # (B, window, 1)
        
        # Additional cost for misclassifying the minority class
        minority_mask = (gt[:,:,1] == 1).float() # (B, window)
        minority_mask = minority_mask.unsqueeze(-1) # (B, window, 1)
        loss = loss * (mask + penalty_cost * minority_mask)
   
        return loss.sum() / mask.sum()

    def _evaluation_metrics(self, output, gt):
        """Generate precision, recall, and f1 score.

        Args:
            output: (B, window, 1)   # prob class
            gt (inference):   (B, window, 3)   # one hot
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
        avg_test_f1, avg_test_loss, avg_test_prec, avg_test_recall = 0.0, 0.0, 0.0, 0.0      
        self.model.eval()
        with torch.no_grad():
            for _ in tqdm(range(self.test_n_batch), desc=f"test data"):
                test_data = next(self.test_dl) 
                test_gt = test_data['gt'] # (B, window, 3)
                
                test_input = {}    
                for idx, body_name in test_data['idx_feats'].items():
                    # (BS, window, 3)
                    test_input[body_name[0]] = test_data[body_name[0]]
                    
                    if not self.preload_gpu:
                        test_input[body_name[0]] = test_input[body_name[0]].to(self.device)

                test_input['event'] = test_data['event'] # (bs)
                
                test_pred = self.model(test_input) # (B, window, 1)
                
                prec, recall, f1 = self._evaluation_metrics(test_pred, 
                                                        test_gt.to(self.device))
                test_loss = self._loss_func(test_pred, test_gt.to(self.device))
                
                avg_test_f1 += f1
                avg_test_loss += test_loss
                avg_test_prec += prec
                avg_test_recall += recall
                
            avg_test_f1 /= self.test_n_batch
            avg_test_loss /= self.test_n_batch
            avg_test_prec /= self.test_n_batch
            avg_test_recall /= self.test_n_batch
            
            avg_test_f1 = round(avg_test_f1.item(), 4)
            avg_test_prec = round(avg_test_prec.item(), 4)
            avg_test_recall = round(avg_test_recall.item(), 4)
            avg_test_loss = avg_test_loss.item()
            
            MYLOGGER.info(f"avg_test_loss: {avg_test_loss}")
            MYLOGGER.info(f"avg_test_f1: {avg_test_f1}")
            MYLOGGER.info(f"avg_test_prec: {avg_test_prec}")
            MYLOGGER.info(f"avg_test_recall: {avg_test_recall}")
            
            if self.use_wandb:
                wandb.run.summary[f'best_test_f1_from_val_{base} / step'] = [avg_test_f1, step]
                wandb.run.summary[f'best_test_prec_from_val_{base} / step'] = [avg_test_prec, step]
                wandb.run.summary[f'best_test_recall_from_val_{base} / step'] = [avg_test_recall, step]
                wandb.run.summary[f'best_test_loss_from_val_{base} / step'] = [avg_test_loss, step]

    def train(self):
        best_val_f1 = 0
        best_val_prec = 0
        best_val_recall = 0
        best_val_loss = float('inf')

        for step_idx in tqdm(range(0, self.train_num_steps), desc="Train"):
            self.model.train()
            
            #* training part -----------------------------------------------------------------------
            train_data = next(self.train_dl)
            
            # print(train_data.keys())
            # print(train_data['idx_feats'].keys())
            # print(len(train_data['idx_feats']))
            # print(train_data['lowerback_acc'].shape)
            # print(len(train_data['event']))
            # print(len(train_data['event'][0]))
            # exit(0)
            
            train_gt = train_data['gt'] # (B, window, 3) one-hot        
            train_input = {}    
            for idx, body_name in train_data['idx_feats'].items():
                # (BS, window, 3)
                train_input[body_name[0]] = train_data[body_name[0]]
                
                if not self.preload_gpu:
                    train_input[body_name[0]] = train_input[body_name[0]].to(self.device)

            # train_input['event'] = [list(i) for i in zip(*train_data['event'])]
            # train_input['event'] = [list(tup) for tup in train_data['event']] # (window, bs)
            
            train_input['event'] = train_data['event']
            
            train_pred = self.model(train_input) # (B,window,1)
            
            
            train_loss = self._loss_func(train_pred, train_gt.to(self.device), train_mode=True)
            train_loss /= self.grad_accum_step
            train_loss.backward()
            
            # check gradients
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).\
                                        to(self.device) for p in parameters]), 2.0)
            nan_exist = False
            if torch.isnan(total_norm):
                MYLOGGER.warning('NaN gradients. Skipping to next data...')
                torch.cuda.empty_cache()
                nan_exist = True
            
            if (step_idx + 1) % self.grad_accum_step == 0:
                if nan_exist:
                    continue
                if self.max_grad_norm is not None:
                    # parameters = [p for p in self.model.parameters() if p.grad is not None]
                    # print("before clipping")
                    # print(parameters[0].shape)
                    # print(parameters[0])
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    # parameters = [p for p in self.model.parameters() if p.grad is not None]
  
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
                avg_val_f1, avg_val_loss, avg_val_prec, avg_val_recall = 0.0, 0.0, 0.0, 0.0
                
                self.model.eval()
                with torch.no_grad():
                    for _ in tqdm(range(self.val_n_batch), desc=f"Validation at epoch {cur_epoch}"):
                        
                        val_data = next(self.val_dl)

                        val_gt = val_data['gt'] # (B, window, 3)
                        
                        val_input = {}    
                        for idx, body_name in val_data['idx_feats'].items():
                            # (BS, window, 3)
                            val_input[body_name[0]] = val_data[body_name[0]]
                            
                            if not self.preload_gpu:
                                val_input[body_name[0]] = val_input[body_name[0]].to(self.device)

                        val_input['event'] = val_data['event'] # (bs)
                        
                        val_pred = self.model(val_input) # (B, window, 1)
                        
                        prec, recall, f1 = self._evaluation_metrics(val_pred, 
                                                                    val_gt.to(self.device))
                        val_loss = self._loss_func(val_pred, val_gt.to(self.device))
                        
                        avg_val_f1 += f1
                        avg_val_loss += val_loss
                        avg_val_prec += prec
                        avg_val_recall += recall
                        
                    avg_val_f1 /= self.val_n_batch
                    avg_val_loss /= self.val_n_batch
                    avg_val_prec /= self.val_n_batch
                    avg_val_recall /= self.val_n_batch
                    
                    if self.use_wandb:
                        log_dict = {
                            "Val/avg_val_loss": avg_val_loss.item(),
                            "Val/avg_val_f1": avg_val_f1.item(),
                            "Val/avg_val_prec": avg_val_prec.item(),
                            "Val/avg_val_recall": avg_val_recall.item(),
                            # "Val/pr_auc": pr_auc,
                        }
                        wandb.log(log_dict, step=step_idx+1)
        
                    MYLOGGER.info(f"avg_val_loss: {avg_val_loss.item():4f}")
                    MYLOGGER.info(f"avg_val_f1: {avg_val_f1.item():4f}")
                    MYLOGGER.info(f"avg_val_prec: {avg_val_prec.item():4f}")
                    MYLOGGER.info(f"avg_val_recall: {avg_val_recall.item():4f}")

                # Log learning rate
                if self.use_wandb:
                    wandb.log({'learning_rate': self.scheduler_optim.get_last_lr()[0]}, 
                              step=step_idx+1)

                if self.save_best_model and avg_val_f1 > best_val_f1:
                    best_val_f1 = avg_val_f1
                    tmp = f"{avg_val_f1.item():4f}"
                    if self.use_wandb:
                        wandb.run.summary['best_val_f1 / step'] = [tmp, step_idx+1]
                    self._save_model(step_idx + 1, base='f1', best=True)
                    self._eval_test_data(step=step_idx + 1, base='f1')

                if self.save_best_model and avg_val_prec > best_val_prec:
                    best_val_prec = avg_val_prec
                    tmp = f"{avg_val_prec.item():4f}"
                    if self.use_wandb:
                        wandb.run.summary['best_val_prec / step'] = [tmp, step_idx+1]
                    self._save_model(step_idx + 1, base='prec', best=True)
                    self._eval_test_data(step=step_idx + 1, base='prec')
                    
                if self.save_best_model and avg_val_recall > best_val_recall:
                    best_val_recall = avg_val_recall
                    tmp = f"{avg_val_recall.item():4f}"
                    if self.use_wandb:
                        wandb.run.summary['best_val_recall / step'] = [tmp, step_idx+1]
                    self._save_model(step_idx + 1, base='recall', best=True)
                    self._eval_test_data(step=step_idx + 1, base='recall')
                    
                if self.save_best_model and avg_val_loss < best_val_loss:
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
