import pickle
import pandas as pd
import numpy as np
import h5py; import hdf5plugin
import os
import torch
import torch.nn as nn
import torch.utils.data as D
import tqdm

from utils import device, correlation_score, exponential_linspace_int, count_parameters, CITESEQ_CODING_GENES, CITESEQ_CONSTANT_GENES
from model import ModelWrapper
import architectures
import losses
from trainer import Trainer
from datasets import METADATA, SPLIT_INTERVALS, TOP_DIR_NAME, H5Dataset

class CiteDataset():
    def __init__(self, split: str):
        """
        dataset for dimensionality reduced inference on ATAC-seq

        Parameters
        ----------
        split : str
            which split to use. must be one of `['train', 'val', 'test', 'all']`
        """
        assert split in ['train', 'val', 'test', 'all']
        
        inputs_file = os.path.join(TOP_DIR_NAME, 'data', 'train_cite_inputs.h5')
        assert os.path.isfile(inputs_file)
        inputs_h5 = h5py.File(inputs_file, 'r')[os.path.split(inputs_file)[1].split('.')[0]]
        
        # create correct split       
        np.random.seed(0)  # to ensure same train, val, test splits every time
        self.length = len(inputs_h5['block0_values'])
        idxs = np.random.permutation(self.length)
        start, stop = SPLIT_INTERVALS[split]
        start, stop = int(start * self.length), int(stop * self.length)
        idxs = idxs[start: stop]
        np.random.seed()  # re-random the seed
        
        self.length = len(idxs)
        assert self.length != 0
                
        inputs_coding = np.load('data/train_cite_inputs_coding.npy')
        inputs_pca = np.load('data/train_cite_inputs_pca.npy')
        assert len(inputs_coding) == len(inputs_pca)
        
        targets_file = os.path.join(TOP_DIR_NAME, 'data', 'train_cite_targets.h5')
        assert os.path.isfile(targets_file)
        targets = np.asarray(h5py.File(targets_file, 'r')[os.path.split(targets_file)[1].split('.')[0]]['block0_values'])
        assert len(inputs_coding) == len(targets), 'inputs and targets arent same size??'
                
        x_tensor = torch.tensor(np.concatenate((inputs_coding[idxs], inputs_pca[idxs]), axis=1))
        y_tensor = torch.tensor(targets[idxs])
        
        del inputs_h5, inputs_coding, inputs_pca, targets
        
        self.d = D.TensorDataset(x_tensor, y_tensor)
                
    def __len__(self):
        return self.length
    
    def get_dataloader(self, batch_size: int, pin_memory=True, num_workers=0, shuffle=True):
        return D.DataLoader(self.d, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)

class CiteModel(ModelWrapper):
    def __init__(self, model_name, in_dim, out_dim, depth):        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.pca = pickle.load(open('pkls/cite_4000_pca.pkl', 'rb'))
        cite_keys = list(pd.read_hdf('data/train_cite_inputs.h5', start=0, stop=1).keys())
        self.coding_idxs = [i for i in range(len(cite_keys)) if cite_keys[i] in CITESEQ_CODING_GENES]
        self.other_idxs = [i for i in range(len(cite_keys)) if (cite_keys[i] not in CITESEQ_CODING_GENES and cite_keys[i] not in CITESEQ_CONSTANT_GENES)]
        
        layer_dims = exponential_linspace_int(start=self.in_dim, end=self.out_dim, num=depth + 1, divisible_by=1)
        self.model = []
        for i in range(depth):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            if i == depth // 2:
                self.model.append(nn.BatchNorm1d(in_dim))
                self.model.append(nn.Dropout(0.05))
            self.model.append(nn.Linear(in_dim, out_dim))
            self.model.append(nn.ReLU())
        self.model.pop()  # no final activation function
        self.model = nn.Sequential(*self.model)
        
        super(CiteModel, self).__init__(self.model, model_name)
    
    def infer(self, 
              x: torch.tensor, 
              day: torch.tensor):
        with torch.no_grad():
            if x.__class__ == torch.Tensor:
                x = x.cpu().detach().numpy()
            coding = x[:, self.coding_idxs]
            other = self.pca.transform(x[:, self.other_idxs])
            inputs = np.concatenate((coding, other), axis=1)
            pred = self.model(torch.tensor(inputs).to(device))
            return pred
    
    def loss(self, 
             x: torch.tensor, 
             day: torch.tensor,
             y: torch.tensor):
        pred = self.model(x) 
        loss  = losses.negative_correlation_loss(pred, y)
        return loss
    
    def eval_err(self, 
                 x: torch.tensor, 
                 day: torch.tensor,
                 y: torch.tensor):
        with torch.no_grad():
            pred = self.model(x) 
            error = losses.negative_correlation_loss(pred, y).item()
            loss = error
        return error, loss

class CiteTwoHeads(ModelWrapper):
    def __init__(self, model_name, in_dim, out_dim, hidden_dim, coding_head_length, other_head_length, body_length, body_type):  
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        self.coding_head_length = coding_head_length
        self.other_head_length = other_head_length
        self.body_length = body_length
          
        # self.pca = pickle.load(open('pkls/cite_4000_pca.pkl', 'rb'))
        cite_keys = list(pd.read_hdf('data/train_cite_inputs.h5', start=0, stop=1).keys())
        self.coding_idxs = [i for i in range(len(cite_keys)) if cite_keys[i] in CITESEQ_CODING_GENES]
        self.other_idxs = [i for i in range(len(cite_keys)) if (cite_keys[i] not in CITESEQ_CODING_GENES and cite_keys[i] not in CITESEQ_CONSTANT_GENES)]
          
        # head for handling the coding genes        
        coding_head_layer_dims = exponential_linspace_int(start=len(self.coding_idxs), end=hidden_dim, num=self.coding_head_length + 1, divisible_by=1)
        self.coding_head = []
        for i in range(self.coding_head_length):
            in_dim = coding_head_layer_dims[i]
            out_dim = coding_head_layer_dims[i + 1]
            self.coding_head.append(nn.Linear(in_dim, out_dim))
            self.coding_head.append(nn.ReLU())
        self.coding_head.append(nn.BatchNorm1d(hidden_dim))
        self.coding_head.append(nn.Dropout(0.05))
        self.coding_head = nn.Sequential(*self.coding_head)

        # head for handling the non-coding genes        
        other_head_layer_dims = exponential_linspace_int(start=4000, end=hidden_dim, num=self.other_head_length + 1, divisible_by=1)
        self.other_head = []
        for i in range(self.other_head_length):
            in_dim = other_head_layer_dims[i]
            out_dim = other_head_layer_dims[i + 1]
            self.other_head.append(nn.Linear(in_dim, out_dim))
            self.other_head.append(nn.ReLU())
        self.other_head.append(nn.BatchNorm1d(hidden_dim))
        self.other_head.append(nn.Dropout(0.05))
        self.other_head = nn.Sequential(*self.other_head)

        # body for combining the coding genes with the output of the head
        if body_type == 'linear':
            body_layer_dims = exponential_linspace_int(start=2 * hidden_dim, end=self.out_dim, num=self.body_length + 1, divisible_by=1)
            self.body = []
            for i in range(self.body_length):
                in_dim = body_layer_dims[i]
                out_dim = body_layer_dims[i + 1]
                self.body.append(nn.Linear(in_dim, out_dim))
                self.body.append(nn.ReLU())
            self.body = nn.Sequential(*self.body)
        elif body_type == 'transformer':
            n_chan = 128
            self.body = nn.Sequential(
                nn.Unflatten(1, torch.Size([1, 2 * hidden_dim])),
                nn.Conv1d(1, n_chan, 1, 1),
                nn.Sequential(*[architectures.TransformerBlock(n_chan, 4, 0.025 if i == self.body_length // 2 else 0.0) for i in range(self.body_length)]),
                nn.Conv1d(n_chan, 1, 1, 1),
                nn.Flatten(1, 2),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                nn.Linear(2 * hidden_dim, 140),
            )
        
        super(CiteTwoHeads, self).__init__({'coding_head': self.coding_head, 'other_head': self.other_head, 'body': self.body}, model_name)
        
    def infer(self, 
            x: torch.tensor):
        with torch.no_grad():
            if x.__class__ == torch.Tensor:
                x = x.cpu().detach().numpy()
            coding = x[:, self.coding_idxs]
            other = self.pca.transform(x[:, self.other_idxs])
            h1 = self.coding_head(torch.tensor(coding).to(device))
            h2 = self.other_head(torch.tensor(other).to(device))
            pred = self.body(torch.cat((h1, h2), dim=1))
            return pred
    
    def loss(self, 
             x: torch.tensor, 
             y: torch.tensor):
        h1 = self.coding_head(x[:, :len(self.coding_idxs)])
        h2 = self.other_head(x[:, len(self.coding_idxs):])
        pred = self.body(torch.cat((h1, h2), dim=1))
        loss  = losses.negative_correlation_loss(pred, y)
        return loss
    
    def eval_err(self, 
                 x: torch.tensor, 
                 y: torch.tensor):
        with torch.no_grad():
            h1 = self.coding_head(x[:, :len(self.coding_idxs)])
            h2 = self.other_head(x[:, len(self.coding_idxs):])
            pred = self.body(torch.cat((h1, h2), dim=1))            
            error = losses.negative_correlation_loss(pred, y).item()
            loss = error
        return error, loss
        

if __name__ == '__main__':
    # ------------------------------------- hyperparameters -------------------------------------------------

    model_name = 'cite_two_heads'
    batch_size = 128

    initial_lr = 0.02
    lr_decay_period = 20
    lr_decay_gamma = 0.5
    weight_decay = 0.0004

    num_epochs = 50
    eval_every = 4
    patience = 3
    num_tries = 4

    # --------------------------------------------------------------------------------------------------------

    # model = CiteModel(model_name, in_dim=4110, out_dim=140, depth=8).load_checkpoint()
    model = CiteTwoHeads(model_name, in_dim=4110, out_dim=140, hidden_dim=512, 
                        coding_head_length=6, other_head_length=8, body_length=8, body_type='linear')
    
    print('preparing datasets')  
    train_dataset = CiteDataset('train')
    val_dataset = CiteDataset('val')
    train_dataloader = train_dataset.get_dataloader(batch_size)
    val_dataloader = val_dataset.get_dataloader(batch_size)
    trainer = Trainer(model, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
    trainer.train(num_epochs, eval_every, patience, num_tries)
    
    test_dataset = CiteDataset('test')
    test_dataloader = test_dataset.get_dataloader(1)
    avg = 0.0
    model.eval()
    with torch.no_grad():
        for x, day, y in tqdm.tqdm(test_dataloader):
            out = model.model(x)
            avg += correlation_score(out.cpu().numpy(), y.numpy())
    avg /= len(test_dataset)
    print('avg correlation score on test data: {}'.format(avg))
