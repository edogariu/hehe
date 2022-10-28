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
    def __init__(self, idxs_to_use: List[int]):
        """
        dataset for dimensionality reduced inference on ATAC-seq

        Parameters
        ----------
        idxs_to_use : List[int]
            which indices to use in the dataset
        """
        self.idxs_to_use = idxs_to_use
        
        self.length = len(idxs_to_use)
        assert self.length != 0
                
        inputs_var = np.load('data/train_multi_inputs_var.npy')
        inputs_pca = np.load('data/train_multi_inputs_pca.npy')
        
        # targets = ss.load_npz('data_sparse/train_multi_targets_sparse.npz').toarray()
        targets = np.load('data/train_multi_targets_pca.npy') #if split == 'train' else ss.load_npz('data_sparse/train_multi_targets_sparse.npz')[idxs].toarray()

        self.inputs = np.concatenate((inputs_var, inputs_pca), axis=1)
        self.targets = targets
        
    def __len__(self):
        return self.length
    
    def get_dataloader(self, split: str, batch_size: int, pin_memory=True, num_workers=0, shuffle=True, drop_last=True):
        
        assert split in ['train', 'val', 'test', 'all']
        
        # create correct split       
        np.random.seed(0)  # to ensure same train, val, test splits every time
        length = self.length
        random_idxs = np.random.permutation(length)
        start, stop = SPLIT_INTERVALS[split]
        start, stop = int(start * length), int(stop * length)
        idxs = self.idxs_to_use[random_idxs[start: stop]]
        np.random.seed()  # re-random the seed

        x_tensor = torch.tensor(self.inputs[idxs])
        y_tensor = torch.tensor(self.targets[idxs])
                
        d = D.TensorDataset(x_tensor, y_tensor)

        return D.DataLoader(d, batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=num_workers)

class MultiModel(ModelWrapper):
    def __init__(self, model_name, num_var_features_to_use, num_pca_features_to_use, hidden_dim, num_target_features_to_use, pyramid_depth, body_depth, dropout=0.25):
        self.in_dim = num_var_features_to_use + num_pca_features_to_use
        self.out_dim = num_target_features_to_use
        
        self.num_var_features_to_use = num_var_features_to_use
        self.num_pca_features_to_use = num_pca_features_to_use
        
        assert self.num_var_features_to_use + self.num_pca_features_to_use == self.in_dim
        
        self.in_pca = pickle.load(open('pkls/multi_4000_pca.pkl', 'rb'))
        self.out_pca = pickle.load(open('pkls/multi_targets_pca.pkl', 'rb'))

        self.multi_var_idxs = np.load('pkls/multi_var_idxs.npy')
        self.multi_pca_idxs = self.multi_var_idxs[4000:]
        self.multi_var_idxs = self.multi_var_idxs[:self.num_var_features_to_use]
        self.cite_var_idxs = np.load('pkls/cite_var_idxs.npy')[:-3418]
        
        self.in_mul = torch.tensor(self.in_pca.components_.T[:, :self.num_pca_features_to_use]).float().to(device)
        self.out_mul = torch.tensor(self.out_pca.components_[:self.out_dim]).float().to(device)
        self.out_mean = torch.tensor(self.out_pca.mean_).to(device)
        self.in_mul.requires_grad = False
        self.out_mul.requires_grad = False
        self.out_mean.requires_grad = False

        # self.final_means = np.load('pkls/last_3418_multi_means.npy')
        
        pyramid_layer_dims = exponential_linspace_int(start=self.in_dim, end=hidden_dim, num=pyramid_depth + 1, divisible_by=1)
        # pyramid_layer_dims = [self.in_dim // (3 ** i) for i in range(depth + 1)]
        pyramid_layers = []
        cat_dim = self.in_dim
        for i in range(pyramid_depth):
            in_dim = pyramid_layer_dims[i]
            out_dim = pyramid_layer_dims[i + 1]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            cat_dim += out_dim
            pyramid_layers.append(nn.Sequential(*layer))
        pyramid_layers = nn.ModuleList(pyramid_layers)

        body_layer_dims = exponential_linspace_int(start=cat_dim, end=self.out_dim, num=body_depth + 1, divisible_by=1)
        body_layers = [nn.BatchNorm1d(cat_dim), nn.Dropout(dropout)]
        for i in range(body_depth):
            in_dim = body_layer_dims[i]
            out_dim = body_layer_dims[i + 1]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            body_layers.extend(layer)
        body_layers.pop()  # remove last ReLU
        body = nn.Sequential(*body_layers)

        self.model = architectures.FPN(pyramid_layers, body)
        super(MultiModel, self).__init__(self.model, model_name)
    
    def loss(self, 
             x: torch.tensor, 
             y: torch.tensor):
        x = torch.cat((x[:, :self.num_var_features_to_use], x[:, 4000:4000 + self.num_pca_features_to_use]), dim=1)
        # y = y[:, self.cite_var_idxs]
        y = y[:, :self.out_dim]
        
        out = self.model(x)
        loss = F.mse_loss(out, y)
        # pred = torch.mm(out, self.out_mul) + self.out_mean
        # loss = losses.negative_correlation_loss(pred, y)
        return loss
    
    def eval_err(self, 
                 x: torch.tensor, 
                 y: torch.tensor):
        with torch.no_grad():
            x = torch.cat((x[:, :self.num_var_features_to_use], x[:, 4000:4000 + self.num_pca_features_to_use]), dim=1)
            # y = y[:, self.cite_var_idxs].cpu().numpy()
            y = y[:, :self.out_dim]

            out = self.model(x)
            loss = F.mse_loss(out, y).item()
            error = loss
            # pred = torch.mm(out, self.out_mul) + self.out_mean
            # error = -correlation_score(out.cpu().numpy(), self.out_pca.transform(y)[:, :self.out_dim])  # compare in pca world
            # loss = -correlation_score(pred.cpu().numpy(), y)  # compare in real world
        return error, loss    
    
    def infer(self, x):
        """
        Infers on entire 228k dim multi vector.
        """
        with torch.no_grad():
            var_in = x[:, self.multi_var_idxs]
            pca_in = torch.mm(x[:, self.multi_pca_idxs], self.in_mul)
            inputs = torch.cat((var_in, pca_in), dim=1)
            out = self.model(inputs)
            pred = torch.mm(out, self.out_mul) + self.out_mean
            return pred



if __name__ == '__main__':
    # ------------------------------------- hyperparameters -------------------------------------------------

    model_name = 'multi_pca_out_huge'
    batch_size = 48

    initial_lr = 0.025
    lr_decay_period = 40
    lr_decay_gamma = 0.7
    weight_decay = 0.0002

    num_epochs = 200
    eval_every = 4
    patience = 3
    num_tries = 4

    # --------------------------------------------------------------------------------------------------------

    print('preparing datasets')
    idxs = get_train_idxs('multi')
    dataset = MultiDataset(idxs)
    train_dataloader = dataset.get_dataloader('train', batch_size)
    val_dataloader = dataset.get_dataloader('val', batch_size)
    
    print('training')
    model = MultiModel(model_name, 
                       num_var_features_to_use=4000, num_pca_features_to_use=4000, num_target_features_to_use=7000, 
                       hidden_dim=512, pyramid_depth=3, body_depth=3)
    # print(model); exit(0)
    trainer = Trainer(model, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
    trainer.train(num_epochs, eval_every, patience, num_tries)
    