import pickle
import os
import h5py; import hdf5plugin
import pandas as pd
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as D

import architectures
from utils import get_train_idxs, count_parameters, device, correlation_score, exponential_linspace_int, TOP_DIR_NAME, CITESEQ_CODING_GENES, CITESEQ_CONSTANT_GENES, METADATA
from model import ModelWrapper
import losses
from trainer import Trainer

CHECKPOINT_FOLDER = os.path.join(TOP_DIR_NAME, 'checkpoints', 'cite_ensemble')

SPLIT_INTERVALS = {'train': (0, 0.85),  # intervals for each split
                   'val': (0.85, 1.0),
                   'test': (0.85, 1.0),
                   'all': (0, 1.0)}

class CiteDataset():
    def __init__(self, split: str, idxs_to_use: List[int]):
        """
        dataset for dimensionality reduced inference on CITE-seq

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
                
        inputs_coding = np.load('data/train_cite_inputs_coding.npy')
        inputs_pca = np.load('data/train_cite_inputs_pca.npy')
        assert len(inputs_coding) == len(inputs_pca)
        
        targets_file = os.path.join(TOP_DIR_NAME, 'data', 'train_cite_targets.h5')
        assert os.path.isfile(targets_file)
        targets = np.asarray(h5py.File(targets_file, 'r')[os.path.split(targets_file)[1].split('.')[0]]['block0_values'])
        assert len(inputs_coding) == len(targets), 'inputs and targets arent same size??'
                
        x_tensor = torch.tensor(np.concatenate((inputs_coding[idxs], inputs_pca[idxs]), axis=1))
        y_tensor = torch.tensor(targets[idxs])
        
        del inputs_coding, inputs_pca, targets
        
        self.d = D.TensorDataset(x_tensor, y_tensor)
                
    def __len__(self):
        return self.length
    
    def get_dataloader(self, batch_size: int, pin_memory=True, num_workers=0, shuffle=True, drop_last=True):
        return D.DataLoader(self.d, batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=num_workers)
    
class CiteModel(ModelWrapper):
    def __init__(self, model_name, in_dim, hidden_dim, out_dim, depth, dropout=0.05):        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.pca = pickle.load(open('pkls/cite_4000_pca.pkl', 'rb'))
        cite_keys = list(pd.read_hdf('data/train_cite_inputs.h5', start=0, stop=1).keys())
        self.coding_idxs = [i for i in range(len(cite_keys)) if cite_keys[i] in CITESEQ_CODING_GENES]
        self.other_idxs = [i for i in range(len(cite_keys)) if (cite_keys[i] not in CITESEQ_CODING_GENES and cite_keys[i] not in CITESEQ_CONSTANT_GENES)]
        
        pyramid_layer_dims = exponential_linspace_int(start=self.in_dim, end=hidden_dim, num=depth + 1, divisible_by=1)
        pyramid_layers = []
        cat_dim = self.in_dim
        for i in range(depth):
            in_dim = pyramid_layer_dims[i]
            out_dim = pyramid_layer_dims[i + 1]
            layer = [nn.Linear(in_dim, out_dim), nn.ReLU()]
            cat_dim += out_dim
            pyramid_layers.append(nn.Sequential(*layer))
        pyramid_layers = nn.ModuleList(pyramid_layers)
        body = nn.Sequential(nn.BatchNorm1d(cat_dim), nn.Dropout(dropout), nn.Linear(cat_dim, cat_dim // 3), nn.ReLU(), nn.Linear(cat_dim // 3, self.out_dim))
        
        self.model = architectures.FPN(pyramid_layers, body)
        super(CiteModel, self).__init__(self.model, model_name, checkpoint_folder=CHECKPOINT_FOLDER)
    
    def infer(self, 
              x: torch.tensor):
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
             y: torch.tensor):
        pred = self.model(x) 
        loss  = losses.negative_correlation_loss(pred, y)
        return loss
    
    def eval_err(self, 
                 x: torch.tensor, 
                 y: torch.tensor):
        with torch.no_grad():
            pred = self.model(x) 
            error = losses.negative_correlation_loss(pred, y).item()
            loss = error
        return error, loss    
    
def find_model_name(cell_id):
    cell_info = METADATA.loc[cell_id]
    cell_type = cell_info['cell_type']
    day = cell_info['day']
    if day == 7: day = 4
    if cell_type in ['MasP', 'BP', 'MkP', 'MoP']:
        cell_type = 'rest'
    return '{}_day{}_cell{}'.format('cite', day, cell_type)

class CiteInferencer():
    def __init__(self):
        self.models = {}
        for day in [2, 3, 4]:
            for cell_type in ['HSC', 'EryP', 'NeuP', 'rest']:
                model_name = 'cite_day{}_cell{}'.format(day, cell_type)
                self.models[model_name] = CiteModel(model_name, in_dim=4110, hidden_dim=256, out_dim=140, depth=6).load_checkpoint().eval()
    
    def infer(self, x, cell_id):
        model = self.models[find_model_name(cell_id)]
        return model.infer(x)

def train(day, cell_type):
    # ------------------------------------- hyperparameters -------------------------------------------------

    # day = 3
    # donor = 32606
    # cell_type = 'HSC'

    model_name = 'cite_day{}_cell{}'.format(day, cell_type if type(cell_type) == str else 'rest')
    batch_size = 128
    
    days = [day]
    cell_types = [cell_type] if type(cell_type) == str else cell_type

    initial_lr = 0.02
    lr_decay_period = 40
    lr_decay_gamma = 0.7
    weight_decay = 0.0002

    num_epochs = 120
    eval_every = 6
    patience = 5
    num_tries = 5

    # --------------------------------------------------------------------------------------------------------

    model = CiteModel(model_name, in_dim=4110, hidden_dim=256, out_dim=140, depth=6)
    
    print('preparing datasets')  
    train_dataset = CiteDataset('train', get_train_idxs('cite', days=days, cell_types=cell_types))
    val_dataset = CiteDataset('val', get_train_idxs('cite', cell_types=cell_types))
    train_dataloader = train_dataset.get_dataloader(batch_size, drop_last=True)
    val_dataloader = val_dataset.get_dataloader(batch_size, drop_last=False)
    trainer = Trainer(model, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
    trainer.train(num_epochs, eval_every, patience, num_tries)
    
    # test_dataset = CiteDataset('test', idxs)
    # test_dataloader = test_dataset.get_dataloader(1)
    # avg = 0.0
    # model.eval()
    # with torch.no_grad():
    #     for x, y in tqdm.tqdm(test_dataloader):
    #         out = model.model(x)
    #         avg += correlation_score(out.cpu().numpy(), y.numpy())
    # avg /= len(test_dataset)
    # print('avg correlation score on test data: {}'.format(avg))
    
if __name__ == '__main__':
    for day in [2, 3, 4]:
        for cell_type in ['HSC', 'EryP', 'NeuP', ['MasP', 'BP', 'MkP', 'MoP']]:
            train(day, cell_type)
