from typing import Dict, Union
import torch
import torch.nn as nn
import torch.utils.data as D
import os
import pickle
import numpy as np

from utils import exponential_linspace_int, count_parameters, TOP_DIR_NAME
from datasets import SparseDataset, H5Dataset
from model import Model
from trainer import Trainer

"""
WHAT IF WE INFERENCED BY CHROMOSOME???????
chr1, ..., chr22, chrx, chrx  --- 24 models
"""

RNA_CHROMOSOME_LENS = {'chr17': 1359, 'chr1': 2276, 'chr2': 1575, 'chrx': 713, 'chr22': 570, 'chr16': 1114, 'chr12': 1295, 'chr11': 1259, 'chr19': 1544, 'chr7': 1081, 'chr14': 795, 'chr10': 881, 'chr13': 433, 'chr3': 1337, 'chr6': 1197, 'chr4': 906, 'chr15': 835, 'chr5': 1057, 'chr20': 569, 'chr8': 921, 'chr9': 856, 'chr18': 424, 'chr21': 283, 'chry': 22, 'chrmt': 13, 'n/a': 103}
DNA_CHROMOSOME_LENS = {'chr17': 7089, 'chr1': 14568, 'chr2': 13377, 'chrx': 3409, 'chr22': 3194, 'chr16': 5375, 'chr12': 8034, 'chr11': 7458, 'chr19': 5562, 'chr7': 8118, 'chr14': 5099, 'chr10': 7076, 'chr13': 3680, 'chr3': 11385, 'chr4': 7170, 'chr15': 5479, 'chr5': 8442, 'chr6': 9512, 'chr20': 3844, 'chr8': 7001, 'chr9': 6203, 'chr18': 3375, 'chr21': 1828, 'chry': 72}

class SillyDNA2RNA(nn.Module):
    def __init__(self, 
                 chrom: str,
                 depth: int,
                 width_factor: float,
                 ):
        super(SillyDNA2RNA, self).__init__()

        assert chrom in RNA_CHROMOSOME_LENS

        self.in_dim = DNA_CHROMOSOME_LENS[chrom]
        self.out_dim = RNA_CHROMOSOME_LENS[chrom]

        if chrom not in DNA_CHROMOSOME_LENS:
            self.constant_output = True
            self.out = nn.parameter.Parameter(torch.zeros(self.out_dim))
            self.out.requires_grad = True
        else:
            self.constant_output = False
            l = self.in_dim
            model_dims = exponential_linspace_int(l, int(width_factor * l), depth)
            model_dims.append(self.out_dim)
            model = []
            for i in range(len(model_dims) - 1):
                in_dim = model_dims[i]
                out_dim = model_dims[i + 1]
                model.append(nn.Linear(in_dim, out_dim))
                model.append(nn.ReLU())
            model.pop() # remove last relu
            self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x) if not self.constant_output else self.out
        return out

class SillySparseDataset(SparseDataset):                              
    def __init__(self, split: str, mode: str, num_genes_to_use=-1, n_data=1000000000, days=[2, 3, 4, 7]):
        super().__init__(split, mode, num_genes_to_use, n_data, days)
        with open('pkls/chrom_dna_map.pkl', 'rb') as f:
            self.chrom_dna_map = pickle.load(f)
        with open('pkls/chrom_rna_map.pkl', 'rb') as f:
            self.chrom_rna_map = pickle.load(f)
        self.day_tensor = torch.tensor([self.metadata.iloc[index]['day'] for index in self.idxs])
            
    def get_dataloader(self, batch_size: int, chrom: str, shuffle=True, pin_memory=True, num_workers=0):
        assert chrom in RNA_CHROMOSOME_LENS

        # if chrom not in self.chrom_dna_map:
        #     inputs = torch.zeros(len(self.idxs))
        # else:
        dna_idxs = np.array(self.chrom_dna_map[chrom])
        inputs = torch.tensor(self.inputs_npz[self.idxs][:, dna_idxs].toarray())

        rna_idxs = np.array(self.chrom_rna_map[chrom])
        dataset = D.TensorDataset(inputs, self.day_tensor, torch.tensor(self.targets_npz[self.idxs][:, rna_idxs].toarray())) 
        return D.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)

class SillyH5Dataset(H5Dataset):    
    """
    FIX THSI FOR THE CHROMS NOT IN DNA
    """                          
    def __init__(self, split: str, mode: str, num_genes_to_use=-1, n_data=1000000000, days=[2, 3, 4, 7]):
        super().__init__(split, mode, num_genes_to_use, n_data, days)
        with open('pkls/chrom_dna_map.pkl', 'rb') as f:
            self.chrom_dna_map = pickle.load(f)
        with open('pkls/chrom_rna_map.pkl', 'rb') as f:
            self.chrom_rna_map = pickle.load(f)
        self.day_tensor = torch.tensor([self.metadata.iloc[index]['day'] for index in self.idxs])

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        index = self.idxs[index]
        day = self.metadata.iloc[index]['day']
        inputs = (self.inputs_h5[index][self.dna_idxs], day)  # could add more to inputs here
        targets = self.targets_h5[index][self.rna_idxs]
        
        return (*inputs, targets)
            
    def get_dataloader(self, batch_size: int, chrom: str, shuffle=True, pin_memory=True, num_workers=0):
        assert chrom in RNA_CHROMOSOME_LENS
        self.dna_idxs = self.chrom_dna_map[chrom]
        self.rna_idxs = self.chrom_rna_map[chrom]
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)

class SillyModel(Model):
    def __init__(self, models: Union[nn.Module, Dict[str, nn.Module]], model_name=None):
        super().__init__(models, model_name)
    
    def load_checkpoint(self):
        for k in self._models:
            model_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'chromosome', 'models', '{}.pth'.format(k))
            opt_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'chromosome', 'optimizers', '{}.pth'.format(k))
            
            assert os.path.isfile(model_filename) and os.path.isfile(opt_filename)

            self._models[k].load_state_dict(torch.load(model_filename), strict=True)
            if self._optimizers:
                self._optimizers[k].load_state_dict(torch.load(opt_filename))
    
    def save_checkpoint(self):
        for k in self._models:
            model_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'chromosome', 'models', '{}.pth'.format(k))
            opt_filename = os.path.join(TOP_DIR_NAME, 'checkpoints', 'chromosome', 'optimizers', '{}.pth'.format(k))

            torch.save(self._models[k].state_dict(), model_filename)
            if self._optimizers:
                torch.save(self._optimizers[k].state_dict(), opt_filename)

def run():
    # ------------------------------------- hyperparameters -------------------------------------------------

    batch_size = 128

    initial_lr = 0.04
    lr_decay_period = 4
    lr_decay_gamma = 0.6
    weight_decay = 0.0001

    num_epochs = 20
    eval_every = 2
    patience = 3
    num_tries = 4

    # --------------------------------------------------------------------------------------------------------

    print('preparing datasets')
    train_dataset = SillySparseDataset('train', 'multi')
    val_dataset = SillySparseDataset('val', 'multi')

    for chrom in RNA_CHROMOSOME_LENS.keys():
        print('TRAINGING FOR {}'.format(chrom))
        model = SillyModel(SillyDNA2RNA(chrom, 3, 0.5), chrom)
        train_dataloader = train_dataset.get_dataloader(batch_size, chrom)
        val_dataloader = val_dataset.get_dataloader(batch_size, chrom)
        trainer = Trainer(model, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
        trainer.train(num_epochs, eval_every, patience, num_tries)

if __name__ == '__main__':
    run()
