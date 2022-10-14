import torch.utils.data as D
import scipy.sparse as ss
import h5py; import hdf5plugin
import os
import pandas as pd
import numpy as np

TOP_DIR_NAME = os.path.dirname(os.path.abspath(__file__))

METADATA = pd.read_csv(os.path.join(TOP_DIR_NAME, 'data', 'metadata.csv')) 
METADATA.set_index('cell_id', inplace=True) # index metadata by cell id

SPLIT_INTERVALS = {'train': (0, 0.8),  # intervals for each split
                   'val': (0.8, 0.95),
                   'test': (0.95, 1.0)}

class H5Dataset(D.Dataset):
    """
    To construct dataloader from original `.h5` files
    """
    def __init__(self, split: str, mode: str, n_data=1e9, days=[2, 3, 4, 7]):
        """
        Creates torch.utils.data.Dataset from the original `.h5` files.

        Parameters
        ----------
        split : str
            which split to use. must be one of `['train', 'val', 'test']`
        mode : str
            which mode to get data from. must be one of `['multi', 'cite']`
        n_data : int
            number of data points to use
        days : List[int]
            which day to draw data from. must be a subset of `[2, 3, 4, 7]`
        """
        super(D.Dataset, self).__init__()
        
        assert split in ['train', 'val', 'test']
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
                    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        index = self.idxs[index]
        day = self.metadata.iloc[index]['day']
        inputs = (self.inputs_h5[index], day)  # could add more to inputs here
        targets = self.targets_h5[index]
        
        return (inputs, targets)
            
    def get_dataloader(self, batch_size: int, shuffle=True):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True)
    
class SparseDataset(D.Dataset):
    """
    To construct dataloader from sparse CSR `.npz` files (much faster `__next__()`)
    """
    def __init__(self, split: str, mode: str, n_data=1e9, days=[2, 3, 4, 7]):
        """
        Creates torch.utils.data.Dataset from the sparse CSR `.npz` files.

        Parameters
        ----------
        split : str
            which split to use. must be one of `['train', 'val', 'test']`
        mode : str
            which mode to get data from. must be one of `['multi', 'cite']`
        n_data : int
            number of data points to use
        days : List[int]
            which day to draw data from. must be a subset of `[2, 3, 4, 7]`
        """
        super(D.Dataset, self).__init__()
        
        assert split in ['train', 'val', 'test']
        assert mode in ['multi', 'cite']
        
        self.split = split
        self.mode = mode
        self.days = days
        
        inputs_file = os.path.join(TOP_DIR_NAME, 'data_sparse', f'train_{mode}_inputs_sparse.npz')
        assert os.path.isfile(inputs_file)
        self.inputs_npz = ss.load_npz(inputs_file)
        
        # prepare matching metadata, such as `day`, `donor`, `cell_type`, `technology`
        ids = np.array(h5py.File(os.path.join(TOP_DIR_NAME, 'data', f'train_{mode}_inputs.h5'), 'r')[f'train_{mode}_inputs']['axis1'], dtype=str)
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
                    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        index = self.idxs[index]
        day = self.metadata.iloc[index]['day']
        inputs = (self.inputs_npz[index].toarray()[0], day)  # could add more to inputs here
        targets = self.targets_npz[index].toarray()[0]
        
        return (inputs, targets)
    
    def get_dataloader(self, batch_size: int, shuffle=True):
        return D.DataLoader(self, batch_size, shuffle=shuffle, drop_last=True)

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
    