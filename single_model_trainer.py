from typing import Dict, Callable
import numpy as np
import torch
import torch.utils.data as D
import tqdm
import os

from datasets import TOP_DIR_NAME
from models import count_parameters

class Trainer():
    def __init__(self, 
                 model : torch.nn.Module, 
                 model_name : str,
                 loss_fn : Callable,
                 error_fn : Callable,
                 train_dataloader: D.DataLoader,
                 val_dataloader: D.DataLoader,
                 initial_lr: float, 
                 lr_decay_period: int, 
                 lr_decay_gamma: float, 
                 weight_decay: float,
                 ):
        """
        Trainer object to train a model. Uses Adam optimizer, StepLR learning rate scheduler, and a patience algorithm.

        Parameters
        ------------
        model : torch.nn.Module
            model to train
        model_name : str
            name to call model when saving checkpoints
        loss_fn : Callable
            callable function to calculate loss for a training example. 
            Signature should be `loss_fn(x, day, y) = loss`, where `loss` is a scalar tensor. Must be differentiable
        error_fn : Callable
            callable function to calculate error for a training example. 
            Signature should be `error_fn(x, day, y) = error`, where `error` is a float. Need not be differentiable
        train_dataloader : D.DataLoader
            dataloader for training data
        val_dataloader : D.DataLoader
            dataloader for validation data. If this is `None`, we do not validate
        initial_lr : float
            learning rate to start with
        lr_decay_period: int 
            how many epochs between each decay step
        lr_decay_gamma: float
            size of each decay step
        weight_decay: float
            l2 regularization
        """
        
        self.model = model
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.error_fn = error_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if val_dataloader is None:
            print('No validation dataloader provided. Skipping validation.')

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('Using {} for training'.format(self.device))

        # prep model and optimizer and scheduler
        self.model.train().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_period, gamma=lr_decay_gamma)

        # prep statistics
        self.train_losses = {}
        self.val_errors = {}
        self.best_val_err = (0, float('inf'))  # epoch num and value of best validation loss

    def train_one_epoch(self, epoch_num):
        self.model.train()
        print()
        print('-------------------------------------------------------------')
        print('------------------  TRAIN - EPOCH NUM {}  -------------------'.format(epoch_num))
        print('-------------------------------------------------------------')
        
        avg_loss = 0.0
        i = 0
        for (x, day), y in tqdm.tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            x = x.to(self.device); day = day.to(self.device); y = y.to(self.device)
            loss = self.loss_fn(x, day, y)
            loss.backward()
            avg_loss += loss.item()
            self.optimizer.step()
            i += 1
        self.scheduler.step()
        avg_loss = avg_loss / i
        self.train_losses[epoch_num] = avg_loss
        print('avg batch training loss for epoch {}: {}'.format(epoch_num, round(avg_loss, 6)))

    def eval(self, epoch_num):
        self.model.eval()
        print()
        print('-------------------------------------------------------------')
        print('-------------------  VAL - EPOCH NUM {}  -------------------'.format(epoch_num))
        print('-------------------------------------------------------------')
        
        with torch.no_grad():
            errs = []
            for (x, day), y in tqdm.tqdm(self.val_dataloader):
                x = x.to(self.device); day = day.to(self.device); y = y.to(self.device)
                
                error = self.error_fn(x, day, y)
                errs.append(error)

        avg_err = np.mean(errs)
        self.val_errors[epoch_num] = avg_err
        print('avg validation error for epoch {}: {}'.format(epoch_num, round(avg_err, 6)))
        return avg_err

    def train(self, num_epochs, eval_every, patience, num_tries):
        """
        Train the model. Applies patience -- every `eval_every` epochs, we eval on validation data using a metric (corner distance) thats not the loss function. 
        We expect the model to improve in this metric as we train: if after `patience` validation steps the model has still not improved, we reset to the best previous checkpoint.
        We attempt this for `num_tries` attempts before we terminate training early.

        Parameters
        ---------------
        num_epochs : int
            number of epochs to train for
        eval_every : int
            interval (measured in epochs) between valiation evaluations
        patience : int
            number of evaluations without improvement before we reset to best checkpoint
        num_tries : int
            number of checkpoint resets before early stopping

        Returns 
        --------------
        model : torch.nn.Module
            the trained model
        """
        print('Training the following model for {} epochs:\n\n{}'.format(num_epochs, str(self.model) + ' with {} parameters'.format(count_parameters(self.model))))
        patience_counter = 0
        tries_counter = 0
        for e in range(num_epochs):
            try:
                self.train_one_epoch(e)
                if e % eval_every == 0:
                    if self.val_dataloader:
                        val_err = self.eval(e)
                        if val_err < self.best_val_err[1]:  # measure whether our model is improving
                            self.best_val_err = (e, val_err)
                            patience_counter = 0
                            self.save_checkpoint()
                            print('Saved checkpoint for epoch num {}'.format(e))
                        else:
                            patience_counter += 1
                            print('Patience {} hit'.format(patience_counter))
                            if patience_counter >= patience:  # if our model has not improved after `patience` evaluations, reset to best checkpoint
                                tries_counter += 1
                                patience_counter = 0
                                self.load_checkpoint()
                                print('Loaded checkpoint from epoch num {}'.format(self.best_val_err[0]))
                                print('Try {} hit'.format(tries_counter))
                                if tries_counter >= num_tries:  # if our model has reset to best checkpoint `num_tries` times, we are done
                                    print('Stopping training!')
                                    break
                    else:
                        self.save_checkpoint()
                        print('Saved checkpoint for epoch num {}'.format(e))
            except KeyboardInterrupt:
                print('Catching keyboard interrupt!!!')
                self.finish_up(e)
                exit(0)
        self.finish_up(num_epochs)
        return self.model

    def finish_up(self, e):
        val_err = self.eval(e) if self.val_dataloader else -float('inf')
        if val_err < self.best_val_err[1]:
            self.save_checkpoint()
            print('Saved checkpoint at the end of training!')

        # k = np.array(list(self.train_losses.keys()))
        # v = np.array(list(self.train_losses.values()))
        # np.save(TOP_DIR_NAME + '/plots/train_losses.npy', np.stack([k, v], axis=0))

        # k = np.array(list(self.val_errors.keys()))
        # v = np.array(list(self.val_errors.values()))
        # np.save(TOP_DIR_NAME + '/plots/val_errors.npy', np.stack([k, v], axis=0))
        
        print('\nDone :)')
    
    def load_checkpoint(self):
        model_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'models', '{}.pth'.format(self.model_name))
        opt_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'optimizers', '{}.pth'.format(self.model_name))
        
        assert os.path.isfile(model_filename) and os.path.isfile(opt_filename)

        self.model.load_state_dict(torch.load(model_filename))
        self.optimizer.load_state_dict(torch.load(opt_filename))
    
    def save_checkpoint(self):
        model_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'models', '{}.pth'.format(self.model_name))
        opt_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'optimizers', '{}.pth'.format(self.model_name))

        torch.save(self.model.state_dict(), model_filename)
        torch.save(self.optimizer.state_dict(), opt_filename)
        