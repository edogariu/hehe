from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Model

class JEPA(Model):
    """
    a class to handle joint embeddings, possibly with predictor between them
    
    in_encoder and out_decoder are necessary. the rest are optional.
    """
    def __init__(self, 
                 models: nn.Module | Dict[str, nn.Module],
                 model_name = 'Joint Embedding (Predictive) Architecture'):
        assert models.__contains__('in_encoder') and models.__contains__('out_decoder') and models['in_encoder'] is not None and models['out_decoder'] is not None
        super(JEPA, self).__init__(models, model_name)
        
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
