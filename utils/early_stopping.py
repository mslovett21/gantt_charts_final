import torch
import numpy as np

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    """
    def __init__(self, path, patience=4, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'early_stopping_vgg16model.pth'   
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0   
    
    def save_checkpoint(self, val_loss, model):
        """
        saves the current best version of the model if there is decrease in validation loss
        """
        torch.save(model.state_dict(), self.path+"early_stopping_checkpoint.pth")
        self.vall_loss_min = val_loss
        