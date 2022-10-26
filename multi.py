import pickle
import scipy.sparse as ss
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
from typing import List

import architectures
import losses
from utils import correlation_score, device, exponential_linspace_int, count_parameters, get_train_idxs, TOP_DIR_NAME, METADATA
from model import ModelWrapper
from trainer import Trainer

SPLIT_INTERVALS = {'train': (0, 0.85),  # intervals for each split
                   'val': (0.85, 1.0),
                   'test': (0.85, 1.0),
                   'all': (0, 1.0)}

class MultiDataset():
    def __init__(self, split: str, idxs_to_use: List[int]):
        """
        dataset for dimensionality reduced inference on ATAC-seq

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
                
        inputs_var = np.load('data/train_multi_inputs_var.npy')[idxs]
        inputs_pca = np.load('data/train_multi_inputs_pca.npy')[idxs]
        
        targets = ss.load_npz('data_sparse/train_multi_targets_sparse.npz')[idxs].toarray()
        # targets = np.load('data/train_multi_targets_pca.npy')[idxs] if split == 'train' else ss.load_npz('data_sparse/train_multi_targets_sparse.npz')[idxs].toarray()
                
        x_tensor = torch.tensor(np.concatenate((inputs_var, inputs_pca), axis=1))
        y_tensor = torch.tensor(targets)
        
        del inputs_var, inputs_pca, targets
        
        self.d = D.TensorDataset(x_tensor, y_tensor)
                
    def __len__(self):
        return self.length
    
    def get_dataloader(self, batch_size: int, pin_memory=True, num_workers=0, shuffle=True, drop_last=True):
        return D.DataLoader(self.d, batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=num_workers)

class MultiModel(ModelWrapper):
    def __init__(self, model_name, num_var_features_to_use, num_pca_features_to_use, hidden_dim, num_target_features_to_use, depth, dropout=0.25):
        self.in_dim = num_var_features_to_use + num_pca_features_to_use
        self.out_dim = num_target_features_to_use
        
        self.num_var_features_to_use = num_var_features_to_use
        self.num_pca_features_to_use = num_pca_features_to_use
        
        assert self.num_var_features_to_use + self.num_pca_features_to_use == self.in_dim
        
        self.out_pca = pickle.load(open('pkls/multi_targets_pca.pkl', 'rb'))
        self.cite_var_idxs = np.load('pkls/cite_var_idxs.npy')[:-3418]
        
        self.out_mul = torch.tensor(self.out_pca.components_[:self.out_dim]).float().to(device)
        self.pca_mean = torch.tensor(self.out_pca.mean_).to(device)
        self.out_mul.requires_grad = False
        self.pca_mean.requires_grad = False
        
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
        super(MultiModel, self).__init__(self.model, model_name)
    
    def loss(self, 
             x: torch.tensor, 
             y: torch.tensor):
        x = torch.cat((x[:, :self.num_var_features_to_use], x[:, 4000:4000 + self.num_pca_features_to_use]), dim=1)
        y = y[:, self.cite_var_idxs]
        
        out = self.model(x)
        pred = torch.mm(out, self.out_mul) + self.pca_mean
        loss = losses.negative_correlation_loss(pred, y)
        return loss
    
    def eval_err(self, 
                 x: torch.tensor, 
                 y: torch.tensor):
        with torch.no_grad():
            x = torch.cat((x[:, :self.num_var_features_to_use], x[:, 4000:4000 + self.num_pca_features_to_use]), dim=1)
            y = y[:, self.cite_var_idxs].cpu().numpy()

            out = self.model(x)
            pred = torch.mm(out, self.out_mul) + self.pca_mean
            error = -correlation_score(out.cpu().numpy(), self.out_pca.transform(y)[:, :self.out_dim])  # compare in pca world
            loss = -correlation_score(pred.cpu().numpy(), y)  # compare in real world
        return error, loss    

if __name__ == '__main__':
    # ------------------------------------- hyperparameters -------------------------------------------------

    model_name = 'multi_pca_out'
    batch_size = 256

    initial_lr = 0.016
    lr_decay_period = 15
    lr_decay_gamma = 0.7
    weight_decay = 0.0002

    num_epochs = 100
    eval_every = 4
    patience = 3
    num_tries = 4

    # --------------------------------------------------------------------------------------------------------

    print('preparing datasets')
    idxs = get_train_idxs('multi')
    train_dataset = MultiDataset('train', idxs)
    val_dataset = MultiDataset('val', idxs)
    train_dataloader = train_dataset.get_dataloader(batch_size)
    val_dataloader = val_dataset.get_dataloader(batch_size)
    
    print('training')
    model = MultiModel(model_name, 
                       num_var_features_to_use=1000, num_pca_features_to_use=1000, num_target_features_to_use=1600, 
                       hidden_dim=512, depth=8)
    # print(model); exit(0)
    trainer = Trainer(model, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
    trainer.train(num_epochs, eval_every, patience, num_tries)
    