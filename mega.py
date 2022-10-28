import pickle
import scipy.sparse as ss
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from typing import List, Tuple
import librosa

import architectures
import losses
from utils import correlation_score, device, exponential_linspace_int, count_parameters, get_train_idxs, TOP_DIR_NAME, METADATA
from model import ModelWrapper
from trainer import Trainer

SPLIT_INTERVALS = {'train': (0, 0.85),  # intervals for each split
                   'val': (0.85, 1.0),
                   'test': (0.85, 1.0),
                   'all': (0, 1.0)}

class MegaDataset(D.Dataset):
    _pca = None
    _inputs = None
    _targets = None
    
    def __init__(self, split: str, mode: str, idxs_to_use: List[int]):
        """
        dataset for pca + mel-spectrogram-ified inference

        Parameters
        ----------
        mode : str
            whether we are using multi or cite datasets. must be one of `['multi', 'cite']`
        idxs_to_use : List[int]
            which indices to use in the dataset
        """
        
        assert split in ['train', 'val', 'test', 'all']
        
        assert mode in ['multi', 'cite']
        self.mode = mode
        
        if not MegaDataset._pca is not None: 
            MegaDataset._pca = np.load(f'data/train_{mode}_inputs_pca.npy')[idxs_to_use] if mode == 'multi' else np.concatenate((np.load(f'data/train_{mode}_inputs_coding.npy'), np.load(f'data/train_{mode}_inputs_pca.npy')), axis=-1)[idxs_to_use]
        if not MegaDataset._inputs is not None: 
            MegaDataset._inputs = pd.read_hdf('data/train_cite_inputs.h5').values[idxs_to_use] # self.inputs = ss.load_npz(f'data_sparse/train_{mode}_inputs_sparse.npz')[idxs_to_use]
        if not MegaDataset._targets is not None: 
            MegaDataset._targets = np.load(f'data/train_{mode}_targets_pca.npy')[idxs_to_use] if mode == 'multi' else pd.read_hdf(f'data/train_{mode}_targets.h5').values[idxs_to_use]
        # self.targets = ss.load_npz(f'data_sparse/train_{mode}_targets_sparse.npz')[idxs_to_use]
        
        # create correct split       
        np.random.seed(0)  # to ensure same train, val, test splits every time
        length = len(idxs_to_use)
        random_idxs = np.random.permutation(length)
        start, stop = SPLIT_INTERVALS[split]
        start, stop = int(start * length), int(stop * length)
        idxs = random_idxs[start: stop]
        np.random.seed()  # re-random the seed
        
        self.idxs = idxs
        self.length = len(idxs)
        
        assert MegaDataset._inputs.shape[0] == MegaDataset._targets.shape[0] and MegaDataset._inputs.shape[0] == MegaDataset._pca.shape[0]
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        index = self.idxs[index]
        pca = MegaDataset._pca[index]
        in_seq = MegaDataset._inputs[index] # in_seq = self.inputs[index].toarray()[0]  # could add more to inputs here
        mel = librosa.feature.melspectrogram(y=in_seq, sr=len(in_seq), hop_length=1800 if self.mode == 'multi' else 173)  # hop lengths to ensure (128 x 128) spectrograms
        if self.mode == 'cite': mel = librosa.power_to_db(mel)
        targets = MegaDataset._targets[index]
        
        return (pca, mel, targets)
                
    def get_dataloader(self, batch_size: int, 
                       pin_memory=True, num_workers=0, shuffle=True, drop_last=True):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=num_workers)

class MelHead(nn.Module):
    def __init__(self, 
                 mel_shape: Tuple[int], out_dim: int,   # mel_shape should be (128, 448) for multi and (128, 44) for cite
                 n_chan: int, tower_depth: int, 
                 pool_every: int, pooling_type: str, pool_size: int, 
                 body_depth: int, body_type: str,
                 dropout: float):
        super().__init__()
        
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
            layer = [nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3))]
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

    def forward(self, x):
        return self.model(x)

class PCAHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        pyramid_layer_dims = exponential_linspace_int(start=self.in_dim, end=hidden_dim, num=depth + 1, divisible_by=1)
        # pyramid_layer_dims = [self.in_dim // (3 ** i) for i in range(depth + 1)]
        pyramid_layers = []
        cat_dim = self.in_dim
        for i in range(depth):
            in_dim = pyramid_layer_dims[i]
            out_dim = pyramid_layer_dims[i + 1]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            cat_dim += out_dim
            pyramid_layers.append(nn.Sequential(*layer))
        pyramid_layers = nn.ModuleList(pyramid_layers)
        mid_dim = (cat_dim + 2 * self.out_dim) // 3
        body = nn.Sequential(nn.BatchNorm1d(cat_dim), nn.Dropout(dropout), 
                             nn.Linear(cat_dim, mid_dim), nn.ReLU(),
                             nn.Linear(mid_dim, mid_dim), nn.ReLU(), 
                             nn.Linear(mid_dim, self.out_dim))
        
        self.model = architectures.FPN(pyramid_layers, body)

    def forward(self, x):
        return self.model(x)    
    
class MegaModel(ModelWrapper):
    def __init__(self, 
                 model_name: str, 
                 hidden_dim: int,
                 out_dim: int,
                 body_depth: int,
                 dropout: float,
                 pca_args,
                 mel_args):
        self.out_dim = out_dim
        
        self.pca_head = PCAHead(**pca_args)
        self.mel_head = MelHead(**mel_args)
        
        pyramid_layer_dims = exponential_linspace_int(start=self.pca_head.out_dim + self.mel_head.out_dim, end=hidden_dim, num=body_depth + 1, divisible_by=1)
        # pyramid_layer_dims = [self.in_dim // (3 ** i) for i in range(depth + 1)]
        pyramid_layers = []
        cat_dim = self.pca_head.out_dim + self.mel_head.out_dim
        for i in range(body_depth):
            in_dim = pyramid_layer_dims[i]
            out_dim = pyramid_layer_dims[i + 1]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            cat_dim += out_dim
            pyramid_layers.append(nn.Sequential(*layer))
        pyramid_layers = nn.ModuleList(pyramid_layers)
        mid_dim = (cat_dim + 2 * self.out_dim) // 3
        body = nn.Sequential(nn.BatchNorm1d(cat_dim), nn.Dropout(dropout), 
                             nn.Linear(cat_dim, mid_dim), nn.ReLU(),
                             nn.Linear(mid_dim, mid_dim), nn.ReLU(), 
                             nn.Linear(mid_dim, self.out_dim))
        
        self.fpn = architectures.FPN(pyramid_layers, body)
                    
        super(MegaModel, self).__init__({f'{model_name}_pca_head': self.pca_head, f'{model_name}_mel_head': self.mel_head, f'{model_name}_fpn': self.fpn}, model_name)
    
    def loss(self, 
             pca: torch.tensor, 
             mel: torch.tensor,
             y: torch.tensor):
        h1 = self.pca_head(pca)
        h2 = self.mel_head(mel)
        cat = torch.cat((h1, h2), dim=1)
        pred = self.fpn(cat)
        loss = losses.negative_correlation_loss(pred, y)
        # loss = F.mse_loss(pred, y[:, :self.out_dim])
        return loss
    
    def eval_err(self, 
                 pca: torch.tensor, 
                mel: torch.tensor,
                y: torch.tensor):
        with torch.no_grad():
            h1 = self.pca_head(pca)
            h2 = self.mel_head(mel)
            cat = torch.cat((h1, h2), dim=1)
            pred = self.fpn(cat)
            loss = -correlation_score(pred.cpu().numpy(), y.cpu().numpy())
            # loss = F.mse_loss(pred, y[:, :self.out_dim]).item()
            error = loss
        return error, loss  

if __name__ == '__main__':
    # ------------------------------------- hyperparameters -------------------------------------------------

    mode = 'cite'; assert mode in ['multi', 'cite']
    model_name = f'mega_{mode}'
    batch_size = 128

    trainer_args = {'initial_lr': 0.03,
                    'lr_decay_period': 10,
                    'lr_decay_gamma': 0.7,
                    'weight_decay': 0.0002}
    train_args = {'num_epochs': 60,
                  'eval_every': 2,
                  'patience': 3,
                  'num_tries': 4}
    
    # multi
    if mode == 'multi':
        fpn_args = {'hidden_dim': 512,
                    'out_dim': 4000,
                    'body_depth': 3,
                    'dropout': 0.15}
        pca_args = {'in_dim': 4000, 
                    'hidden_dim': 512, 
                    'out_dim': 4000, 
                    'depth': 4}
        mel_args = {'mel_shape': (128, 128),
                    'out_dim': 4000,
                    'n_chan': 128,
                    'tower_depth': 8,
                    'body_type': 'linear',
                    'body_depth': 4,
                    'pooling_type': 'max',
                    'pool_every': 2,
                    'pool_size': 2,
                    'dropout': 0.1}
    # cite
    else:
        fpn_args = {'hidden_dim': 512,
                    'out_dim': 140,
                    'body_depth': 6,
                    'dropout': 0.1}
        pca_args = {'in_dim': 4110, 
                    'hidden_dim': 512, 
                    'out_dim': 512, 
                    'depth': 5}
        mel_args = {'mel_shape': (128, 128),
                    'out_dim': 512,
                    'n_chan': 256,
                    'tower_depth': 10,
                    'body_type': 'linear',
                    'body_depth': 4,
                    'pooling_type': 'max',
                    'pool_every': 2,
                    'pool_size': 2,
                    'dropout': 0.1}
        

    # --------------------------------------------------------------------------------------------------------

    print('preparing datasets')
    idxs = get_train_idxs(mode)
    train_dataset = MegaDataset('train', mode, idxs)
    val_dataset = MegaDataset('val', mode, idxs)
    train_dataloader = train_dataset.get_dataloader(batch_size)
    val_dataloader = val_dataset.get_dataloader(batch_size)
    
    print('training')
    model = MegaModel(model_name, **fpn_args, pca_args=pca_args, mel_args=mel_args)
    # model = Baby(model_name, **model_args)
    # print(model); exit(0)
    trainer = Trainer(model, train_dataloader, val_dataloader, **trainer_args)
    trainer.train(**train_args)
    