import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from typing import List
import scipy.sparse as ss

from model import ModelWrapper
import architectures
import losses
from utils import get_train_idxs, exponential_linspace_int, device
from trainer import Trainer

SPLIT_INTERVALS = {'train': (0, 0.85),  # intervals for each split
                   'val': (0.85, 1.0),
                   'test': (0.85, 1.0),
                   'all': (0, 1.0)}

class AutoEncoderDataset():
    def __init__(self, split: str, idxs_to_use: List[int]):
        """
        Parameters
        ----------
        split : str
            which split to use. must be one of `['train', 'val', 'test', 'all']`
        idxs_to_use : List[int]
            which indices to use in the dataset
        """
        assert split in ['train', 'val', 'test', 'all']
        
        # create correct split       
        np.random.seed(0)  # to ensure same train, val, test splits every time
        length = len(idxs_to_use)
        random_idxs = np.random.permutation(length)
        start, stop = SPLIT_INTERVALS[split]
        start, stop = int(start * length), int(stop * length)
        idxs = idxs_to_use[random_idxs[start: stop]]
        np.random.seed()  # re-random the seed
        
        self.length = len(idxs)
        assert self.length != 0
                
        inputs = np.load('data/train_multi_targets_pca.npy')[idxs]
        
        targets = ss.load_npz('data_sparse/train_multi_targets_sparse.npz')[idxs].toarray()
                
        x_tensor = torch.tensor(inputs)
        y_tensor = torch.tensor(targets)
        
        del inputs, targets
        
        self.d = D.TensorDataset(x_tensor, y_tensor)
    
    def __len__(self):
        return self.length
    
    def get_dataloader(self, batch_size: int, pin_memory=True, num_workers=0, shuffle=True, drop_last=True):
        return D.DataLoader(self.d, batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=num_workers)

class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth, end_relu=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        layer_dims = exponential_linspace_int(start=self.in_dim, end=hidden_dim, num=depth + 1, divisible_by=1)
        # layer_dims = [self.in_dim // (3 ** i) for i in range(depth + 1)]
        pyramid_layers = []
        cat_dim = self.in_dim
        for i in range(depth):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            cat_dim += out_dim
            pyramid_layers.append(nn.Sequential(*layer))
        pyramid_layers = nn.ModuleList(pyramid_layers)
        mid_dim = (cat_dim + self.out_dim) // 2
        if end_relu: body = nn.Sequential(nn.BatchNorm1d(cat_dim), nn.Dropout(0.08), nn.Linear(cat_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, self.out_dim), nn.ReLU())
        else: body = nn.Sequential(nn.BatchNorm1d(cat_dim), nn.Dropout(0.08), nn.Linear(cat_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, self.out_dim))
        
        self.model = architectures.FPN(pyramid_layers, body)
    
    def forward(self, x):
        return self.model(x)
    

class AutoEncoder(ModelWrapper):
    # def __init__(self, in_dim, out_dim, num_channels, tower_length, body_length, body_type, pooling_type):
    #     self.out_pca = pickle.load(open('pkls/multi_targets_pca.pkl', 'rb'))
        
    #     self.encoder = architectures.Encoder(in_dim, out_dim, num_channels, tower_length, body_length, body_type, pooling_type)
    #     self.decoder = architectures.Decoder(out_dim, in_dim, num_channels, tower_length, body_length, body_type, 'conv')
        
    #     super().__init__({'encoder': self.encoder, 'decoder': self.decoder}, 'autoencoder')
    
    def __init__(self, in_dim, hidden_dim, out_dim, body_length):
        self.out_pca = pickle.load(open('pkls/multi_targets_pca.pkl', 'rb'))
        self.in_mul = torch.tensor(self.out_pca.components_.T).to(device)
        self.out_mul = torch.tensor(self.out_pca.components_).to(device)
            
        self.encoder = Model(in_dim, hidden_dim, out_dim, body_length).model
        self.decoder = Model(out_dim, hidden_dim, in_dim, body_length).model
        
        super().__init__({'encoder': self.encoder, 'decoder': self.decoder}, 'autoencoder')
    
    def loss(self, x: torch.tensor, y: torch.tensor):
        # y = (y != 0).float()
        enc = self.encoder(x)
        pred = self.decoder(enc)
        pred = torch.mm(pred, self.out_mul)
        loss = losses.negative_correlation_loss(pred, y)
        return loss
    
    def eval_err(self, x: torch.tensor, y: torch.tensor):
        with torch.no_grad():
            # y = (y != 0).float()
            enc = self.encoder(x)
            pred = self.decoder(enc)
            pred = torch.mm(pred, self.out_mul)
            loss = losses.negative_correlation_loss(pred, y).item()
            return loss, loss
        
    def infer(self, x: torch.tensor):
        with torch.no_grad():
            enc = self.encoder(x)
        return enc
    
if __name__ == '__main__':
    # ------------------------------------- hyperparameters -------------------------------------------------

    batch_size = 256

    initial_lr = 0.02
    lr_decay_period = 15
    lr_decay_gamma = 0.5
    weight_decay = 0.0004

    num_epochs = 30
    eval_every = 3
    patience = 3
    num_tries = 4

    # --------------------------------------------------------------------------------------------------------

    print('preparing datasets')
    idxs = get_train_idxs('multi')
    train_dataset = AutoEncoderDataset('train', idxs)
    val_dataset = AutoEncoderDataset('val', idxs)
    train_dataloader = train_dataset.get_dataloader(batch_size)
    val_dataloader = val_dataset.get_dataloader(batch_size)
    
    print('training')
    # model = AutoEncoder(23418, 2000, 256, 2, 6, 'linear', 'attention')
    model = AutoEncoder(4000, 256, 1600, 6)
    trainer = Trainer(model, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
    trainer.train(num_epochs, eval_every, patience, num_tries)
    