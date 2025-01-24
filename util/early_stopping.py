import numpy as np
import torch
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0,mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_max=0
        self.val_loss_min = np.Inf
        self.delta = delta
        self.mode=mode

    def __call__(self, val_loss, model,f1):
        if self.mode == 'max':
            score = f1
        elif self.mode == 'min':
            score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,f1, model)
            self.counter = 0

    def save_checkpoint(self, val_loss,f1, model):
        if self.verbose:
            if self.mode=='min':
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                self.val_loss_min = val_loss
            elif self.mode=="max":
                print(f'Validation f1 increased ({self.f1_max:.6f} --> {f1:.6f}).  Saving model ...')
                self.f1_max=f1
        torch.save(model.state_dict(), '/home/lhl/cross_certification/new_train/r4.2/base_simcse_model.pth')  # 当验证误差下降时，保存模型。