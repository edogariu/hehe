import torch
import torch.utils.data as D
import scipy.sparse as ss
import h5py; import hdf5plugin
import os
import pandas as pd
import numpy as np

from utils import TOP_DIR_NAME

METADATA = pd.read_csv(os.path.join(TOP_DIR_NAME, 'data', 'metadata.csv')) 
METADATA.set_index('cell_id', inplace=True) # index metadata by cell id

SPLIT_INTERVALS = {'train': (0, 0.8),  # intervals for each split
                   'val': (0.8, 0.95),
                   'test': (0.95, 1.0),
                   'all': (0, 1.0)}

class H5Dataset(D.Dataset):
    """
    To construct dataloader from original `.h5` files
    """
    def __init__(self, split: str, mode: str, num_genes_to_use: int, n_data=1e9, days=[2, 3, 4, 7]):
        """
        Creates torch.utils.data.Dataset from the original `.h5` files.

        Parameters
        ----------
        split : str
            which split to use. must be one of `['train', 'val', 'test', 'all']`
        mode : str
            which mode to get data from. must be one of `['multi', 'cite']`
        num_genes_to_use : int
            how many genes from the input mode to use (uses top `num_genes_to_use` with most variance and nonzero entries)
        n_data : int
            number of data points to use
        days : List[int]
            which day to draw data from. must be a subset of `[2, 3, 4, 7]`
        """
        super(D.Dataset, self).__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        assert mode in ['multi', 'cite']
        for d in days: assert d in [2, 3, 4, 7]
        
        self.split = split
        self.mode = mode
        self.days = days
        
        inputs_file = os.path.join(TOP_DIR_NAME, 'data', f'train_{mode}_inputs.h5')
        assert os.path.isfile(inputs_file)
        self.inputs_h5 = h5py.File(inputs_file, 'r')[os.path.split(inputs_file)[1].split('.')[0]]
        
        # prepare matching metadata, such as `day`, `donor`, `cell_type`, `technology`
        ids = np.array(self.inputs_h5['axis1']).astype(str)
        self.metadata = METADATA.loc[ids]
                
        self.inputs_h5 = self.inputs_h5['block0_values']
        
        targets_file = os.path.join(TOP_DIR_NAME, 'data', f'train_{mode}_targets.h5')
        assert os.path.isfile(targets_file)
        self.targets_h5 = h5py.File(targets_file, 'r')[os.path.split(targets_file)[1].split('.')[0]]['block0_values']
        assert len(self.inputs_h5) == len(self.targets_h5), 'inputs and targets arent same size??'
                
        # create correct split       
        np.random.seed(0)  # to ensure same train, val, test splits every time
        self.length = len(self.inputs_h5)
        self.idxs = np.random.permutation(self.length)
        start, stop = SPLIT_INTERVALS[self.split]
        start, stop = int(start * self.length), int(stop * self.length)
        self.idxs = self.idxs[start: stop]
        np.random.seed()  # re-random the seed
        
        # grab only points from the given days
        self.idxs = self.idxs[np.argwhere(np.isin(self.metadata['day'][self.idxs], self.days)).ravel()]
                
        if n_data < self.length:
            self.idxs = self.idxs[:n_data]

        self.length = len(self.idxs)
        assert self.length != 0
        
        self.gene_idxs = np.load('data/{}_best_idxs.npy'.format(mode))[:num_genes_to_use]
                    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        index = self.idxs[index]
        day = self.metadata.iloc[index]['day']
        inputs = (self.inputs_h5[index][self.gene_idxs], day)  # could add more to inputs here
        targets = self.targets_h5[index]
        
        return (*inputs, targets)
            
    def get_dataloader(self, batch_size: int, shuffle=True, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)
    
class SparseDataset(D.Dataset):
    """
    To construct dataloader from sparse CSR `.npz` files (much faster `__next__()`)
    """
    def __init__(self, split: str, mode: str, num_genes_to_use: int, n_data=1e9, days=[2, 3, 4, 7]):
        """
        Creates torch.utils.data.Dataset from the sparse CSR `.npz` files.

        Parameters
        ----------
        split : str
            which split to use. must be one of `['train', 'val', 'test', 'all']`
        mode : str
            which mode to get data from. must be one of `['multi', 'cite']`
        num_genes_to_use : int
            how many genes from the input mode to use (uses top `num_genes_to_use` with most variance and nonzero entries)
        n_data : int
            number of data points to use
        days : List[int]
            which day to draw data from. must be a subset of `[2, 3, 4, 7]`
        """
        super(D.Dataset, self).__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        assert mode in ['multi', 'cite']
        
        self.split = split
        self.mode = mode
        self.days = days
        
        inputs_file = os.path.join(TOP_DIR_NAME, 'data_sparse', f'train_{mode}_inputs_sparse.npz')
        assert os.path.isfile(inputs_file)
        self.inputs_npz = ss.load_npz(inputs_file)
        
        # prepare matching metadata, such as `day`, `donor`, `cell_type`, `technology`
        ids = np.array(h5py.File(os.path.join(TOP_DIR_NAME, 'data', f'train_{mode}_inputs.h5'), 'r')[f'train_{mode}_inputs']['axis1']).astype(str)
        self.metadata = METADATA.loc[ids]
        
        targets_file = os.path.join(TOP_DIR_NAME, 'data_sparse', f'train_{mode}_targets_sparse.npz')
        assert os.path.isfile(targets_file)
        self.targets_npz = ss.load_npz(targets_file)
        assert self.inputs_npz.shape[0] == self.targets_npz.shape[0], 'inputs and targets arent same size??'
        
        # create correct split       
        np.random.seed(0)  # to ensure same train, val, test splits every time
        self.length = self.inputs_npz.shape[0]
        self.idxs = np.random.permutation(self.length)
        start, stop = SPLIT_INTERVALS[self.split]
        start, stop = int(start * self.length), int(stop * self.length)
        self.idxs = self.idxs[start: stop]
        np.random.seed()  # re-random the seed
        
        # grab only points from the given days
        self.idxs = self.idxs[np.argwhere(np.isin(self.metadata['day'][self.idxs], self.days)).ravel()]
        
        if n_data < self.length:
            self.idxs = np.random.permutation(self.length)[:n_data]
        
        self.length = len(self.idxs)
        assert self.length != 0
        
        self.gene_idxs = np.load('data/{}_best_idxs.npy'.format(mode))[:num_genes_to_use]
        self.inputs_npz = self.inputs_npz[:, self.gene_idxs]
                    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        index = self.idxs[index]
        day = self.metadata.iloc[index]['day']
        inputs = (self.inputs_npz[index].toarray()[0], day)  # could add more to inputs here
        targets = self.targets_npz[index].toarray()[0]
        
        return (*inputs, targets)
    
    def get_dataloader(self, batch_size: int, pin_memory=True, num_workers=0, shuffle=True):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

class NaiveDataset(D.Dataset):
    """
    Simply places the entire dataset into a big tensor.
    """
    def __init__(self, split: str, mode: str, num_genes_to_use: int, n_data=1e9, days=[2, 3, 4, 7]):
        """
        Creates torch.utils.data.TensorDataset from the entire dataset.

        Parameters
        ----------
        split : str
            which split to use. must be one of `['train', 'val', 'test', 'all']`
        mode : str
            which mode to get data from. must be one of `['multi', 'cite']`
        num_genes_to_use : int
            how many genes from the input mode to use (uses top `num_genes_to_use` with most variance and nonzero entries)
        n_data : int
            number of data points to use
        days : List[int]
            which day to draw data from. must be a subset of `[2, 3, 4, 7]`
        """
        super(D.Dataset, self).__init__()
        
        assert split in ['train', 'val', 'test', 'all']
        assert mode in ['multi', 'cite']
        
        self.split = split
        self.mode = mode
        self.days = days
        
        inputs_file = os.path.join(TOP_DIR_NAME, 'data_sparse', f'train_{mode}_inputs_sparse.npz')
        assert os.path.isfile(inputs_file)
        self.inputs_npz = ss.load_npz(inputs_file)
        
        # prepare matching metadata, such as `day`, `donor`, `cell_type`, `technology`
        ids = np.array(h5py.File(os.path.join(TOP_DIR_NAME, 'data', f'train_{mode}_inputs.h5'), 'r')[f'train_{mode}_inputs']['axis1']).astype(str)
        self.metadata = METADATA.loc[ids]
        
        targets_file = os.path.join(TOP_DIR_NAME, 'data_sparse', f'train_{mode}_targets_sparse.npz')
        assert os.path.isfile(targets_file)
        self.targets_npz = ss.load_npz(targets_file)
        assert self.inputs_npz.shape[0] == self.targets_npz.shape[0], 'inputs and targets arent same size??'
        
        # create correct split       
        np.random.seed(0)  # to ensure same train, val, test splits every time
        self.length = self.inputs_npz.shape[0]
        self.idxs = np.random.permutation(self.length)
        start, stop = SPLIT_INTERVALS[self.split]
        start, stop = int(start * self.length), int(stop * self.length)
        self.idxs = self.idxs[start: stop]
        np.random.seed()  # re-random the seed
        
        # grab only points from the given days
        self.idxs = self.idxs[np.argwhere(np.isin(self.metadata['day'][self.idxs], self.days)).ravel()]
        
        if n_data < self.length:
            self.idxs = np.random.permutation(self.length)[:n_data]
        
        self.length = len(self.idxs)
        assert self.length != 0

        self.gene_idxs = np.load('data/{}_best_idxs.npy'.format(mode))[:num_genes_to_use]

        x = self.inputs_npz[self.idxs, self.gene_idxs]
        y = self.targets_npz[self.idxs]
        days = [self.metadata.iloc[index]['day'] for index in self.idxs]
        
        x = torch.tensor(x.toarray())
        y = torch.tensor(y.toarray())
        days = torch.tensor(days)
        self.d = D.TensorDataset(x, days, y)
        
        del self.inputs_npz, self.targets_npz, self.gene_idxs
                    
    def __len__(self):
        return self.length
    
    def get_dataloader(self, batch_size: int, pin_memory=True, num_workers=0, shuffle=True):
        return D.DataLoader(self.d, batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin_memory, num_workers=num_workers)

class SubmissionDataset(D.Dataset):
    """
    To construct dataloader from original `.h5` files
    """
    def __init__(self, mode: str, num_genes_to_use: int):
        """
        Creates torch.utils.data.Dataset from the original `.h5` files.

        Parameters
        ----------
        mode : str
            which mode to get data from. must be one of `['multi', 'cite']`
        num_genes_to_use : int
            how many genes from the input mode to use (uses top `num_genes_to_use` with most variance and nonzero entries)
        """
        super(D.Dataset, self).__init__()
        
        assert mode in ['multi', 'cite']
        
        self.mode = mode
        
        inputs_file = os.path.join(TOP_DIR_NAME, 'data', f'test_{mode}_inputs.h5')
        assert os.path.isfile(inputs_file)
        self.inputs_h5 = h5py.File(inputs_file, 'r')[os.path.split(inputs_file)[1].split('.')[0]]
        
        # prepare matching metadata, such as `day`, `donor`, `cell_type`, `technology`
        ids = np.array(self.inputs_h5['axis1']).astype(str)
        self.metadata = METADATA.loc[ids]
                
        self.inputs_h5 = self.inputs_h5['block0_values']
        self.length = len(self.inputs_h5)
        
        self.gene_idxs = np.load('data/{}_best_idxs.npy'.format(mode))[:num_genes_to_use]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        day = self.metadata.iloc[index]['day']
        inputs = (self.inputs_h5[index][self.gene_idxs], day)  # could add more to inputs here
        
        return inputs
            
    def get_dataloader(self, batch_size: int, pin_memory=True, num_workers=0):
        return D.DataLoader(self, batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

if __name__ == '__main__':
    """
    Runs timing test for the two loaders.
    """
    
    method = 'h5'
    # method = 'npz'
    
    assert method in ['h5', 'npz']
    
    split = 'test'
    mode = 'multi'
    batch_size = 16
    
    dataset = H5Dataset(split, mode) if method == 'h5' else SparseDataset(split, mode)
    
    loader = dataset.get_dataloader(batch_size, shuffle=True)
    from time import perf_counter
    s = perf_counter()
    for x, y in loader:
        x, day = x
        assert len(x) == len(y) and len(x) == batch_size
    print(f'{round(1000 * (perf_counter() - s) / len(loader), 2)} ms per batch on average')
    