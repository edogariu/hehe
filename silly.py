from typing import Dict, Union
import torch
import torch.nn as nn
import torch.utils.data as D
import os
import pickle
import numpy as np
import tqdm

from utils import device, exponential_linspace_int, count_parameters, TOP_DIR_NAME
from datasets import SparseDataset, H5Dataset
from model import ModelWrapper
from trainer import Trainer

"""
WHAT IF WE INFERENCED BY CHROMOSOME???????
chr1, ..., chr22, chrx, chry  --- 24 models
"""

RNA_CHROMOSOME_LENS = {'chr17': 1359, 'chr1': 2276, 'chr2': 1575, 'chrx': 713, 'chr22': 570, 'chr16': 1114, 'chr12': 1295, 'chr11': 1259, 'chr19': 1544, 'chr7': 1081, 'chr14': 795, 'chr10': 881, 'chr13': 433, 'chr3': 1337, 'chr6': 1197, 'chr4': 906, 'chr15': 835, 'chr5': 1057, 'chr20': 569, 'chr8': 921, 'chr9': 856, 'chr18': 424, 'chr21': 283, 'chry': 22, 'chrmt': 13, 'n/a': 103}
DNA_CHROMOSOME_LENS = {'chr17': 7089, 'chr1': 14568, 'chr2': 13377, 'chrx': 3409, 'chr22': 3194, 'chr16': 5375, 'chr12': 8034, 'chr11': 7458, 'chr19': 5562, 'chr7': 8118, 'chr14': 5099, 'chr10': 7076, 'chr13': 3680, 'chr3': 11385, 'chr4': 7170, 'chr15': 5479, 'chr5': 8442, 'chr6': 9512, 'chr20': 3844, 'chr8': 7001, 'chr9': 6203, 'chr18': 3375, 'chr21': 1828, 'chry': 72}

class SillyDNA2RNA(nn.Module):
    def __init__(self, 
                 chrom: str,
                 depth: int,
                 width_factor: float,
                 use_bn: bool
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
                if i - 1 == depth // 2:
                    if use_bn: model.append(nn.BatchNorm1d(in_dim))
                    model.append(nn.Dropout(0.04))
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

class SillyModel(ModelWrapper):
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
                
class SillyInferencer():
    def __init__(self):
        self.models = {}
        self.dna_idxs = pickle.load(open('pkls/chrom_dna_map.pkl', 'rb'))
        self.rna_idxs = pickle.load(open('pkls/chrom_rna_map.pkl', 'rb'))
        for chrom in DNA_CHROMOSOME_LENS.keys():
            model = SillyDNA2RNA(chrom, 5, 0.5, True)
            model.load_state_dict(torch.load('checkpoints/chromosome/models/{}_sigmoid.pth'.format(chrom), map_location=device))
            model.eval()
            self.models[chrom] = model
    
    def eval(self):
        for m in self.models.values():
            m.eval()
        return self
    
    def to(self, device: torch.device):
        for m in self.models.values():
            m.to(device)
        return self
    
    def infer(self, x):
        out = torch.zeros((x.shape[0], 23418))
        for chrom in DNA_CHROMOSOME_LENS.keys():
            out[:, self.rna_idxs[chrom]] = torch.sigmoid(self.models[chrom](x[:, self.dna_idxs[chrom]]))
        return out
    
    def infer_on_whole_dataset(self, dataset, batch_size):
        ret = np.zeros((len(dataset), 23418))
        for chrom in self.models.keys():
            print(chrom)
            dl = dataset.get_dataloader(batch_size, idxs_to_use=self.dna_idxs[chrom])
            m = self.models[chrom]
            m.eval().to(device)
            rna_idxs = self.rna_idxs[chrom]
            model_outs = torch.zeros((len(dataset), len(rna_idxs)))
            with torch.no_grad():
                i = 0
                for x in tqdm.tqdm(dl):
                    b = x.shape[0]
                    x = x.to(device)
                    out = m(x)
                    model_outs[i:i + b] = out.cpu()
                    i += b
            ret[:, rna_idxs] = model_outs.detach().numpy()
        return ret
                
        

def run():
    # ------------------------------------- hyperparameters -------------------------------------------------

    model_type = 'sigmoid'  # shoudl be one of ['sigmoid', 'regression']

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
    train_dataset = SillySparseDataset('train', 'multi')
    val_dataset = SillySparseDataset('val', 'multi')
    
    # train_dataset = SillyH5Dataset('train', 'multi')
    # val_dataset = SillyH5Dataset('val', 'multi')

    chroms_to_train = ['!chr1', '!chr10', '!chr11', 'chr12', '!chr13', '!chr14', '!chr15', '!chr16', '!chr17', '!chr18', '!chr19', 
                       '!chr2', '!chr20', '!chr21', 'chr22', 'chr3', '!chr4', '!chr5', '!chr6', '!chr7', '!chr8', '!chr9', '!chrx', '!chry']

    for chrom in chroms_to_train:
        if '!' in chrom: continue  # skip ones we don't want to train
        print('TRAINGING FOR {}'.format(chrom))
        model = SillyModel(SillyDNA2RNA(chrom, 5, 0.5, use_bn=True), '{}_{}'.format(chrom, model_type))
        train_dataloader = train_dataset.get_dataloader(batch_size, chrom)
        val_dataloader = val_dataset.get_dataloader(batch_size, chrom)
        trainer = Trainer(model, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
        trainer.train(num_epochs, eval_every, patience, num_tries)

if __name__ == '__main__':
    run()

    # import pickle
    # import pandas as pd
    # multi_keys = list(pd.read_hdf('data/train_multi_inputs.h5', start=0, stop=1).keys())
    # cite_keys = list(pd.read_hdf('data/train_multi_targets.h5', start=0, stop=1).keys())
    # with open('pkls/partition.pkl', 'rb') as f:
    #     ret = pickle.load(f)
    #     champ = None
    #     best = 0
    #     for r in ret:
    #         if len(r[0]) > 4:
    #             if len(r[1]) > best:
    #                 best = len(r[1])
    #                 champ = r
    #     rna_idxs = np.sort([cite_keys.index(k) for k in champ[0]])
    #     dna_idxs = np.sort([multi_keys.index(k) for k in champ[1]])
    # from datasets import H5Dataset, SparseDataset
    # from model import Model
    # import torch
    # import torch.nn as nn
    # import tqdm
    # from utils import focal_loss, device

    # # ------------------------------------- hyperparameters -------------------------------------------------

    # batch_size = 144

    # initial_lr = 0.02
    # lr_decay_period = 4
    # lr_decay_gamma = 0.5
    # weight_decay = 0.0004

    # num_epochs = 11
    # eval_every = 2
    # patience = 3
    # num_tries = 4

    # # --------------------------------------------------------------------------------------------------------

    # model = nn.Sequential(
    #     nn.Linear(186, 200),
    #     nn.ReLU(),
    #     nn.Linear(200, 50),
    #     nn.ReLU(),
    #     nn.Linear(50, 27)
    # )
    # model.to(device)
    # model.train()

    # train_dataloader = SparseDataset('train', 'multi').get_dataloader(batch_size)
    # val_dataloader = SparseDataset('val', 'multi').get_dataloader(batch_size)

    # optim = torch.optim.Adam(model.parameters(), initial_lr, weight_decay=weight_decay)
    # for _ in range(num_epochs):
    #     avg_loss = 0.0
    #     for x, day, y in tqdm.tqdm(train_dataloader):
    #         optim.zero_grad()
    #         out = torch.sigmoid(model(x[:, dna_idxs].to(device)))
    #         y = y[:, rna_idxs].to(device)
    #         loss = torch.nn.functional.binary_cross_entropy(out, y)
    #         loss.backward()
    #         avg_loss += loss.cpu().item()
    #         optim.step()
    #     avg_loss /= len(train_dataloader)
    #     print(avg_loss)
