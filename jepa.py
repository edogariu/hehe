import os
from typing import Dict
from models import *
import torch
import torch.nn.functional as F
import torch.optim as optim

class JEPA():
    """
    a class to handle joint embeddings, possibly with predictor between them
    
    in_encoder and out_decoder are necessary. the rest are optional.
    """
    def __init__(self, 
                 models: Dict[str, nn.Module],):
        assert models.__contains__('in_encoder') and models.__contains__('out_decoder') and models['in_encoder'] is not None and models['out_decoder'] is not None
        
        self._models = models
        
        self._in_encoder = models['in_encoder']
        self._in_decoder = models['in_decoder'] if models.__contains__('in_decoder') else None
        self._out_encoder = models['out_encoder'] if models.__contains__('out_encoder') else None
        self._out_decoder = models['out_decoder']
        self._predictor = models['predictor'] if models.__contains__('predictor') else None
        
        self._latent_dim = self._in_encoder.out_dim
        self._in_dim = self._in_encoder.in_dim
        self._out_dim = self._out_decoder.out_dim
        
        if self._predictor:
            assert self._in_encoder.out_dim == self._predictor.in_dim and self._predictor.out_dim == self._out_decoder.in_dim, 'inconsistent latent dim'
        else:
            assert self._in_encoder.out_dim == self._out_decoder.in_dim, 'inconsistent latent dim'
        
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
        
    def init_optimizers_and_lr_schedulers(self, 
                                          initial_lrs: Dict[str, float], 
                                          lr_decay_periods: Dict[str, int], 
                                          lr_decay_gammas: Dict[str, float], 
                                          weight_decays: Dict[str, float]):
        """
        Creates optimiziers and learning rate schedulers for each model.

        Parameters
        ----------
        initial_lrs : Dict[str, float]
            learning rate to start with for each model
        lr_decay_periods : Dict[str, int]
            how many epochs between each decay step for each model
        lr_decay_gammas : Dict[str, float]
            size of each decay step for each model
        weight_decays : Dict[str, float]
            l2 regularization for each model
        """
        self._optimizers = {}
        self._lr_schedulers = {}
        for k in self._models.keys():
            self._optimizers[k] = optim.Adam(self._models[k].parameters(), lr=initial_lrs[k], weight_decay=weight_decays[k])
            self._lr_schedulers[k] = optim.lr_scheduler.StepLR(self._optimizers[k], step_size=lr_decay_periods[k], gamma=lr_decay_gammas[k], verbose=True)
    
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
            enc = x
            enc = self._in_encoder(enc, day)  # encode input into latent space 
            if self._predictor:
                enc = self._predictor(enc, day)  # predict latent space output from latent space input
            out = self._out_decoder(enc, day)  # decode output from latent space
            return out
    
    def loss(self, 
             x: torch.tensor, 
             day: torch.tensor,
             y: torch.tensor):
        """
        Loss for a batch of training examples. 
        This is where we choose to define the loss, which is what the training process attempts to minimize. This must be differentiable.
        TODO add a cooler loss function

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
        reconstruction_weight = 0.5
        prediction_weight = 1.0
        
        loss = 0
        enc = self._in_encoder(x, day)
        if self._in_decoder:
            reconstructed = self._in_decoder(enc, day)
            loss += reconstruction_weight * F.mse_loss(reconstructed, x)
        if self._predictor:
            enc = self._predictor(enc, day)
        pred = self._out_decoder(enc, day)
        loss += prediction_weight * F.mse_loss(pred, y)
        return loss
    
    def eval_err(self, 
                 x: torch.tensor, 
                 day: torch.tensor,
                 y: torch.tensor):
        """
        Error for a batch of training examples. 
        This is where we choose to define the error, which is what we want the model to ultimately do best at. This doesn't need to be differentiable.
        TODO add a smarter evaluation metric

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
        out = self.infer(x, day)
        error = F.mse_loss(out, y).item()
        # loss = self.loss(x, day, y)
        return error, error
    
    def __str__(self) -> str:
        s = 'Joint Embedding (Predictive) Architecture with the following models:\n'
        for k in self._models.keys():
            s += '\t{} with {} parameters\n'.format(k, count_parameters(self._models[k]))
        return s
    
    def load_checkpoint(self):
        for k in self._models:
            model_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'models', '{}.pth'.format(k))
            opt_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'optimizers', '{}.pth'.format(k))
            
            assert os.path.isfile(model_filename) and os.path.isfile(opt_filename)

            self._models[k].load_state_dict(torch.load(model_filename))
            self._optimizers[k].load_state_dict(torch.load(opt_filename))
    
    def save_checkpoint(self):
        for k in self._models:
            model_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'models', '{}.pth'.format(k))
            opt_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'optimizers', '{}.pth'.format(k))

            torch.save(self._models[k].state_dict(), model_filename)
            torch.save(self._optimizers[k].state_dict(), opt_filename)
