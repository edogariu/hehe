import pickle
import scipy.sparse as ss
import h5py; import hdf5plugin
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from typing import List, Tuple

import architectures
import losses
from utils import correlation_score, device, exponential_linspace_int, count_parameters, get_train_idxs, TOP_DIR_NAME, METADATA
from model import ModelWrapper
from trainer import Trainer

SPLIT_INTERVALS = {'train': (0, 0.85),  # intervals for each split
                   'val': (0.85, 1.0),
                   'test': (0.85, 1.0),
                   'all': (0, 1.0)}

class MelDataset():
    def __init__(self, mode: str, idxs_to_use: List[int]):
        """
        dataset for mel-spectrogram-ified inference

        Parameters
        ----------
        mode : str
            whether we are using multi or cite datasets. must be one of `['multi', 'cite']`
        idxs_to_use : List[int]
            which indices to use in the dataset
        """
        
        assert mode in ['multi', 'cite']
        
        idxs_to_use = idxs_to_use[:20000]
        
        self.inputs = np.load(f'data/train_{mode}_inputs_mel.npy')[idxs_to_use]
        
        if mode == 'multi':
            self.targets = np.load('data/train_multi_targets_pca.npy')[idxs_to_use]   # for pca reduced cite targets
            # self.targets = ss.load_npz('data_sparse/train_multi_targets_sparse.npz')[idxs_to_use].toarray()  # for entire cite targets
        else:
            targets_file = os.path.join(TOP_DIR_NAME, 'data', f'train_{mode}_targets.h5')
            assert os.path.isfile(targets_file)
            self.targets = np.asarray(h5py.File(targets_file, 'r')[os.path.split(targets_file)[1].split('.')[0]]['block0_values'])[idxs_to_use]
        
        assert self.inputs.shape[0] == self.targets.shape[0]
        self.length = self.inputs.shape[0]
                
    def get_dataloader(self, split: str, batch_size: int, 
                       pin_memory=True, num_workers=0, shuffle=True, drop_last=True):
        
        assert split in ['train', 'val', 'test', 'all']
        
        # create correct split       
        np.random.seed(0)  # to ensure same train, val, test splits every time
        length = self.length
        random_idxs = np.random.permutation(length)
        start, stop = SPLIT_INTERVALS[split]
        start, stop = int(start * length), int(stop * length)
        idxs = random_idxs[start: stop]
        np.random.seed()  # re-random the seed
        
        x = torch.tensor(self.inputs[idxs])
        y = torch.tensor(self.targets[idxs])
        d = D.TensorDataset(x, y)
        
        return D.DataLoader(d, batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=num_workers)

class MelModel(ModelWrapper):
    def __init__(self, 
                 model_name: str, 
                 mel_shape: Tuple[int], out_dim: int,   # mel_shape should be (128, 448) for multi and (128, 44) for cite
                 n_chan: int, tower_depth: int, 
                 pool_every: int, pooling_type: str, pool_size: int, 
                 body_depth: int, body_type: str,
                 dropout: float):
        
        assert body_type in ['linear', 'transformer']
        assert pooling_type in ['max', 'average', 'attention'] 
        
        pools = {'max': architectures.MaxPool2D,
                 'average': architectures.AvgPool2D,
                 'attention': architectures.AttentionPool2D}
        
        self.in_shape = mel_shape
        self.out_dim = out_dim
        
        tower_channels = exponential_linspace_int(start=1, end=n_chan, num=tower_depth + 1)
        tower_layers = [nn.Unflatten(1, (1, mel_shape[0]))]
        for i in range(tower_depth):
            in_chan, out_chan = tower_channels[i: i + 2]
            layer = [nn.Conv2d(in_chan, out_chan, kernel_size=(5, 3))]
            if (i + 1) % pool_every == 0:
                layer.append(pools[pooling_type](out_chan, pool_size))
            layer.append(nn.ReLU())
            tower_layers.extend(layer)
        if body_type == 'linear': 
            tower_layers.append(nn.Flatten(1, 3))
        elif body_type == 'transformer':
            tower_layers.append(nn.Flatten(2, 3))
        self.tower = nn.Sequential(*tower_layers)
        
        # figure out the tower output dimension in the most silly way possible
        test_tensor = torch.zeros(1, *mel_shape)
        tower_out_dim = self.tower(test_tensor).shape[-1]
        del test_tensor
        
        if body_type == 'linear':
            body_layers = [nn.BatchNorm1d(tower_out_dim), nn.Dropout(dropout)]
            body_layer_dims = exponential_linspace_int(start=tower_out_dim, end=self.out_dim, num=body_depth + 1)
            for i in range(body_depth):
                in_dim, out_dim = body_layer_dims[i: i + 2]
                layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
                body_layers.extend(layer)
            body_layers.pop()
            self.body = nn.Sequential(*body_layers)
        elif body_type == 'transformer':
            n_heads = 4
            body_channels = exponential_linspace_int(start=n_chan, end=self.out_dim, num=body_depth + 1, divisible_by=n_heads)
            body_layers = [nn.BatchNorm1d(n_chan), nn.Dropout(dropout)]
            for i in range(body_depth):
                in_chan, out_chan = body_channels[i: i + 2]
                layer = [architectures.TransformerBlock(in_chan, n_heads, 0.075), nn.Conv1d(in_chan, out_chan, 1)]
                body_layers.extend(layer)
            body_layers.extend([nn.Linear(tower_out_dim, tower_out_dim // 2), nn.ReLU(), nn.Linear(tower_out_dim // 2, 1), nn.Flatten(1, 2)])
            self.body = nn.Sequential(*body_layers)
        
        self.model = nn.Sequential(self.tower, self.body)
                    
        super(MelModel, self).__init__(self.model, model_name)
    
    def loss(self, 
             x: torch.tensor, 
             y: torch.tensor):
        pred = self.model(x)
        # loss = losses.negative_correlation_loss(pred, y)
        loss = F.mse_loss(pred, y[:, :4000])
        return loss
    
    def eval_err(self, 
                 x: torch.tensor, 
                 y: torch.tensor):
        with torch.no_grad():
            pred = self.model(x)
            # loss = -correlation_score(pred.cpu().numpy(), y.cpu().numpy())
            loss = F.mse_loss(pred, y[:, :4000]).item()
            error = loss
        return error, loss  
    
    
class Baby(ModelWrapper):
    def __init__(self, model_name: str, mel_shape: Tuple[int], out_dim: int, depth: int):
        layer_dims = exponential_linspace_int(start=mel_shape[0] * mel_shape[1], end=out_dim, num=depth + 1)
        layers = [nn.Flatten(1, 2)]
        for i in range(depth):
            in_dim, out_dim = layer_dims[i: i + 2]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            layers.extend(layer)
        layers.pop()
        model = nn.Sequential(*layers)
        
        super().__init__(model, model_name)

if __name__ == '__main__':
    # ------------------------------------- hyperparameters -------------------------------------------------

    mode = 'multi'
    model_name = f'mel_{mode}'
    batch_size = 128

    trainer_args = {'initial_lr': 0.03,
                    'lr_decay_period': 15,
                    'lr_decay_gamma': 0.7,
                    'weight_decay': 0.0002}
    train_args = {'num_epochs': 300,
                  'eval_every': 3,
                  'patience': 3,
                  'num_tries': 4}

    # model_args = {'mel_shape': (128, 44),
    #               'out_dim': 140,
    #               'n_chan': 64,
    #               'tower_depth': 4,
    #               'body_type': 'linear',
    #               'body_depth': 6,
    #               'pooling_type': 'attention',
    #               'pool_every': 3,
    #               'pool_size': 2,
    #               'dropout': 0.05}
    
    model_args = {'mel_shape': (128, 448),
                  'out_dim': 4000,
                  'n_chan': 128,
                  'tower_depth': 8,
                  'body_type': 'linear',
                  'body_depth': 4,
                  'pooling_type': 'max',
                  'pool_every': 2,
                  'pool_size': 2,
                  'dropout': 0.05}
    
    # model_args = {'mel_shape': (128, 44),
    #               'out_dim': 140,
    #               'depth': 15}

    # --------------------------------------------------------------------------------------------------------

    print('preparing datasets')
    idxs = get_train_idxs(mode)
    dataset = MelDataset(mode, idxs)
    train_dataloader = dataset.get_dataloader('train', batch_size)
    val_dataloader = dataset.get_dataloader('val', batch_size)
    
    print('training')
    model = MelModel(model_name, **model_args)
    # model = Baby(model_name, **model_args)
    # print(model); exit(0)
    trainer = Trainer(model, train_dataloader, val_dataloader, **trainer_args)
    trainer.train(**train_args)
    