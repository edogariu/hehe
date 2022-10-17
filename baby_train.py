from models import *
from datasets import H5Dataset, SparseDataset
from single_model_trainer import Trainer

import pandas as pd
import os
import tqdm

def train():
    print('preparing data')

    multi_df = pd.read_hdf('data/train_multi_targets.h5', start=1000, stop=2000)
    cite_df = pd.read_hdf('data/train_cite_inputs.h5', start=1000, stop=2000)

    multi_keys = list(multi_df.keys())
    cite_keys = list(cite_df.keys())

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------      HYPERPARAMETERS      ---------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------

    METHOD = 'cite'  # change this to decide which model to train
    model_filename = 'rna_to_protein'

    batch_size = 16

    num_epochs = 10
    eval_every = 1
    patience = 1000
    num_tries = 2

    initial_lr = 0.04
    lr_decay_period = 1
    lr_decay_gamma = 0.7
    weight_decay = 0.0001
    dropout = 0.01

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------

    train_dataset = H5Dataset('train', METHOD, days=[2, 3, 4])
    val_dataset = H5Dataset('val', METHOD, days=[2, 3, 4])
    test_dataset = H5Dataset('test', METHOD, days=[2, 3, 4])

    model = Encoder(len(cite_keys), 140, 128, 6, 6, 'enformer', 'attention', output_2d=False)
    device = torch.device('cuda')

    train_dataloader = train_dataset.get_dataloader(batch_size)
    val_dataloader = val_dataset.get_dataloader(batch_size)

    class PredLoss():
        def __init__(self, model):
            self.model = model
        
        def loss(self, x, day, y):
            rna = x
            protein_hat = self.model(rna, day)
            loss = F.mse_loss(protein_hat, y)
            return loss

        def error(self, x, day, y):
            rna = x
            protein_hat = self.model(rna, day)
            loss = F.mse_loss(protein_hat, y)
            return loss.item()

    l = PredLoss(model)
    # model.load_state_dict(torch.load(os.path.join(TOP_DIR_NAME, 'checkpoints', 'models', '{}.pth'.format(model_filename))))
    trainer = Trainer(model, 
                        model_filename, 
                        l.loss, 
                        l.error, 
                        train_dataloader, 
                        None, #val_dataloader, 
                        initial_lr, 
                        lr_decay_period, 
                        lr_decay_gamma, 
                        weight_decay, 
                        grad_accumulation_steps=1)
    trainer.train(num_epochs, eval_every, patience, num_tries)

    print('testing model')
    model.load_state_dict(torch.load(os.path.join(TOP_DIR_NAME, 'checkpoints', 'models', '{}.pth'.format(model_filename))))
    model.eval()
    model.to(device)
    test_dataloader = test_dataset.get_dataloader(1)
    total_err = 0.0
    for (x, day), y in tqdm.tqdm(test_dataloader):
        total_err += l.error(x, day, y)
    print(total_err / len(test_dataloader))

if __name__ == '__main__':
    train()
    