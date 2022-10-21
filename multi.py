import scipy.sparse as ss
import numpy as np
import h5py; import hdf5plugin
import os
import torch
import torch.nn as nn
import torch.utils.data as D

from utils import exponential_linspace_int, count_parameters
from model import Model
from trainer import Trainer
from datasets import METADATA, SPLIT_INTERVALS, TOP_DIR_NAME

class MultiDataset():
    def __init__(self, split: str):
        """
        dataset for dimensionality reduced inference on ATAC-seq

        Parameters
        ----------
        split : str
            which split to use. must be one of `['train', 'val', 'test', 'all']`
        """
        assert split in ['train', 'val', 'test', 'all']
        days = [2, 3, 4, 7]
        
        inputs_file = os.path.join(TOP_DIR_NAME, 'data', 'train_multi_inputs.h5')
        assert os.path.isfile(inputs_file)
        inputs_h5 = h5py.File(inputs_file, 'r')[os.path.split(inputs_file)[1].split('.')[0]]
        
        # prepare matching metadata, such as `day`, `donor`, `cell_type`, `technology`
        ids = np.array(inputs_h5['axis1']).astype(str)
        metadata = METADATA.loc[ids]
                
        # create correct split       
        np.random.seed(0)  # to ensure same train, val, test splits every time
        self.length = len(inputs_h5['block0_values'])
        idxs = np.random.permutation(self.length)
        start, stop = SPLIT_INTERVALS[split]
        start, stop = int(start * self.length), int(stop * self.length)
        idxs = idxs[start: stop]
        np.random.seed()  # re-random the seed
        
        # grab only points from the given days
        idxs = idxs[np.argwhere(np.isin(metadata['day'][idxs], days)).ravel()]  
        self.length = len(idxs)
        assert self.length != 0
                
        inputs_var = np.load('data/train_multi_inputs_var.npy')[idxs]
        inputs_pca = np.load('data/train_multi_inputs_pca.npy')[idxs]
        
        targets = ss.load_npz('data_sparse/train_multi_targets_sparse.npz')[idxs]
                
        x_tensor = torch.tensor(np.concatenate((inputs_var, inputs_pca), axis=1))
        day_tensor = torch.tensor([metadata.iloc[index]['day'] for index in idxs])
        y_tensor = torch.tensor(targets.toarray())
        
        del inputs_h5, inputs_var, inputs_pca, targets
        
        self.d = D.TensorDataset(x_tensor, day_tensor, y_tensor)
                
    def __len__(self):
        return self.length
    
    def get_dataloader(self, batch_size: int, pin_memory=True, num_workers=0, shuffle=True):
        return D.DataLoader(self.d, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)

class MultiModel(nn.Module):
    def __init__(self, in_dim, out_dim, depth):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        layer_dims = [self.in_dim,] * depth
        layer_dims.extend(exponential_linspace_int(start=self.in_dim, end=self.out_dim, num=4, divisible_by=1))
        self.body = []
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            if i == len(layer_dims) // 2:
                self.body.append(nn.BatchNorm1d(in_dim))
                self.body.append(nn.Dropout(0.05))
            self.body.append(nn.Linear(in_dim, out_dim))
            self.body.append(nn.ReLU())
        self.body = nn.Sequential(*self.body)
    
    def forward(self, x):
        return self.body(x)

if __name__ == '__main__':
    # ------------------------------------- hyperparameters -------------------------------------------------

    model_name = 'multi_var_pca'
    batch_size = 144

    initial_lr = 0.02
    lr_decay_period = 4
    lr_decay_gamma = 0.5
    weight_decay = 0.0004

    num_epochs = 11
    eval_every = 2
    patience = 3
    num_tries = 4

    # --------------------------------------------------------------------------------------------------------

    print('preparing datasets')
    train_dataset = MultiDataset('train')
    val_dataset = MultiDataset('val')
    train_dataloader = train_dataset.get_dataloader(batch_size)
    val_dataloader = val_dataset.get_dataloader(batch_size)
    
    model = MultiModel(in_dim=8000, out_dim=23418, depth=5)
    print('{} with {} parameters'.format(model, count_parameters(model)))
    trainer = Trainer(Model(model, model_name), train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
    trainer.train(num_epochs, eval_every, patience, num_tries)
    