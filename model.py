import os
from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import TOP_DIR_NAME, count_parameters, focal_loss, negative_correlation_loss, nonzero_l1_loss, bce_loss

# LOSS_FN = negative_correlation_loss
# LOSS_FN = F.l1_loss
# LOSS_FN = nonzero_l1_loss
LOSS_FN = focal_loss
# LOSS_FN = bce_loss

EVAL_FN = negative_correlation_loss

"""
This class might look real useless right about now. But I promise its practicing good design and maintainability practices!

I made it in order to allow for seamless addition of multiple model ensembles. For example, frameworks that use joint embeddings, or perhaps different models for different days. 
In general, this defines the API for how to interact with the models, and makes it easy to change it to add more complicated things. 

Without changing any method signatures or anything, all one has to do is reimplement `.infer()`, `.loss()`, and `.eval_err()`   :)
For an example of this, look at `jepa.py`
"""

class Model():
    """
    a class to interact with any framework of models
    """
    def __init__(self, 
                 models: Union[nn.Module, Dict[str, nn.Module]],
                 model_name = None):
        self._model_name = model_name if model_name else 'main'  # only used in the __str__() method and as the dict key if training a single nn.Module
        self._models = models if type(models) == dict else {self._model_name: models}

        self._optimizers = None
        self._lr_schedulers = None
        
    def train(self):
        """
        Change models to train mode
        """
        for model in self._models.values():
            model.train()
        return self
        
    def eval(self):
        """
        Change models to evaluation mode
        """
        for model in self._models.values():
            model.eval()
        return self
        
    def to(self, device: torch.device):
        """
        Move models to given device.

        Parameters
        ----------
        device : torch.device
            device
        """
        for model in self._models.values():
            model.to(device)
        return self
        
    def init_optimizer_and_lr_scheduler(self, 
                                        initial_lr: Union[float, Dict[str, float]], 
                                        lr_decay_period: Union[int, Dict[str, int]], 
                                        lr_decay_gamma: Union[float, Dict[str, float]], 
                                        weight_decay: Union[float, Dict[str, float]]):
        """
        Creates optimiziers and learning rate schedulers for each model.

        Parameters
        ----------
        initial_lr : Union[float, Dict[str, float]]
            learning rate to start with for each model
        lr_decay_period : Union[int, Dict[str, int]]
            how many epochs between each decay step for each model
        lr_decay_gamma : Union[float, Dict[str, float]]
            size of each decay step for each model
        weight_decay : Union[float, Dict[str, float]]
            l2 regularization for each model
        """
        if type(initial_lr) == float: initial_lr = {self._model_name: initial_lr}
        if type(lr_decay_period) == int: lr_decay_period = {self._model_name: lr_decay_period}
        if type(lr_decay_gamma) == float: lr_decay_gamma = {self._model_name: lr_decay_gamma}
        if type(weight_decay) == float: weight_decay = {self._model_name: weight_decay}
        
        self._optimizers = {}
        self._lr_schedulers = {}
        for k in self._models.keys():
            self._optimizers[k] = optim.Adam(self._models[k].parameters(), lr=initial_lr[k], weight_decay=weight_decay[k])
            self._lr_schedulers[k] = optim.lr_scheduler.StepLR(self._optimizers[k], step_size=lr_decay_period[k], gamma=lr_decay_gamma[k], verbose=True)
    
    def step_optimizers(self):
        """
        Steps optimizers for each model
        """
        for optimizer in self._optimizers.values():
            optimizer.step()
    
    def zero_grad(self):
        """
        Zeros gradients of optimizers for each model
        """
        for optimizer in self._optimizers.values():
            optimizer.zero_grad()
    
    def step_lr_schedulers(self):
        """
        Steps learning rate schedulers for each model
        """
        for lr_scheduler in self._lr_schedulers.values():
            lr_scheduler.step()
        
    def infer(self, 
              x: torch.tensor, 
              day: torch.tensor):
        """
        Convert input sequence to output sequence

        Parameters
        ----------
        x : torch.tensor
            input sequence of shape `(batch_size, in_seq_len)`
        day : torch.tensor
            day that input measurements were made, of shape `(batch_size,)`

        Returns
        -------
        torch.tensor
            output sequence of shape `(batch_size, out_seq_len)`
        """
        with torch.no_grad():
            pred = self._models[self._model_name](x)  # default assumes only one model in ensemble
            return pred
    
    def loss(self, 
             x: torch.tensor, 
             day: torch.tensor,
             y: torch.tensor):
        """
        Loss for a batch of training examples. 
        This is where we choose to define the loss, which is what the training process attempts to minimize. This must be differentiable.

        Parameters
        ----------
        x : torch.tensor
            input sequence of shape `(batch_size, in_seq_len)`
        day : torch.tensor
            day that input measurements were made, of shape `(batch_size,)`
        y : torch.tensor
            target output sequence of shape `(batch_size, out_seq_len)`
            
        Returns
        ------------
        torch.tensor
            loss
        """
        pred = self._models[self._model_name](x)  # default assumes only one model in ensemble
        loss  = LOSS_FN(pred, y)
        return loss
    
    def eval_err(self, 
                 x: torch.tensor, 
                 day: torch.tensor,
                 y: torch.tensor):
        """
        Error for a batch of training examples. 
        This is where we choose to define the error, which is what we want the model to ultimately do best at. This doesn't need to be differentiable.

        Parameters
        ----------
        x : torch.tensor
            input sequence of shape `(batch_size, in_seq_len)`
        day : torch.tensor
            day that input measurements were made, of shape `(batch_size,)`
        y : torch.tensor
            target output sequence of shape `(batch_size, out_seq_len)`
            
        Returns
        ------------
        float
            error
        float
            loss
        """
        with torch.no_grad():
            pred = self._models[self._model_name](x) # default assumes only one model in ensemble
            error = LOSS_FN(pred, y).item()
            loss = EVAL_FN(pred, y).item()
        return error, loss
    
    def __str__(self) -> str:
        s = '{} with the following parts:\n\n'.format(self._model_name)
        for k in self._models.keys():
            s += str(self._models[k])
            s += '\n\t{} with {} parameters\n'.format(k, count_parameters(self._models[k]))
        return s
    
    def load_checkpoint(self):
        for k in self._models:
            model_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'models', '{}.pth'.format(k))
            opt_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'optimizers', '{}.pth'.format(k))
            
            assert os.path.isfile(model_filename) and os.path.isfile(opt_filename)

            self._models[k].load_state_dict(torch.load(model_filename), strict=True)
            if self._optimizers:
                self._optimizers[k].load_state_dict(torch.load(opt_filename))
    
    def save_checkpoint(self):
        for k in self._models:
            model_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'models', '{}.pth'.format(k))
            opt_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'optimizers', '{}.pth'.format(k))

            torch.save(self._models[k].state_dict(), model_filename)
            if self._optimizers:
                torch.save(self._optimizers[k].state_dict(), opt_filename)
