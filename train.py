import pandas as pd
import torch
import torch.nn.functional as F

import architectures
from datasets import H5Dataset, SparseDataset
from trainer import Trainer
from model import Model

# ----------------------------------------------------------------------------------------------------
# ------------------------------------   CITEseq Training   ------------------------------------------
# ----------------------------------------------------------------------------------------------------

# ------------------------------------- hyperparameters -------------------------------------------------
batch_size = 512

initial_lr = 0.05
lr_decay_period = 10
lr_decay_gamma = 0.5
weight_decay = 0.0001

num_epochs = 30
eval_every = 3
patience = 4
num_tries = 1

model = architectures.RNA2Protein(in_dim=22050, out_dim=140, hidden_dim=2000, head_length=2, body_length=6)
# --------------------------------------------------------------------------------------------------------

train_dataloader = SparseDataset('train', 'cite').get_dataloader(batch_size)
val_dataloader = SparseDataset('val', 'cite').get_dataloader(batch_size)

# model.load_state_dict(torch.load('checkpoints/models/cite_other.pth'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

trainer = Trainer(Model(model, 'rna_to_protein'), train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
trainer.train(num_epochs, eval_every, patience, num_epochs)

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------










# ----------------------------------------------------------------------------------------------------
# ------------------------------------   JEPA Training   ---------------------------------------------
# ----------------------------------------------------------------------------------------------------

# mode = 'cite'
# days = [2, 3, 4, 7]

# # get data
# train_dataset = H5Dataset('train', mode, days=days)
# train_dataloader = train_dataset.get_dataloader(batch_size=8)

# val_dataset = H5Dataset('val', mode, days=days)
# val_dataloader = val_dataset.get_dataloader(batch_size=8)

# # make model
# latent_dim = 2048
# latent_2d = True
# n_chan = 512

# in_encoder = Encoder(22050, latent_dim, n_chan, 6, 11, 'enformer', 'attention', output_2d=latent_2d)
# in_decoder = Decoder(latent_dim, 22050, n_chan, 6, 11, 'enformer', 'conv', input_2d=latent_2d)
# out_decoder = Decoder(latent_dim, 140, n_chan, 6, 11, 'enformer', 'conv', input_2d=latent_2d)
# predictor = LinearCoder([latent_dim, latent_dim], 0.05, input_2d=latent_2d, days=days)
# models = {'in_encoder': in_encoder,
#           'in_decoder': in_decoder,
#           'out_decoder': out_decoder,
#           'predictor': predictor}
# jepa = JEPA(models)

# # train
# initial_lrs = {'in_encoder': 0.04,
#                'in_decoder': 0.04,
#              'out_decoder': 0.04,
#              'predictor': 0.04}
# lr_decay_periods = {'in_encoder': 5,
#                     'in_decoder': 5,
#              'out_decoder': 5,
#              'predictor': 5}
# lr_decay_gammas = {'in_encoder': 0.5,
#                 'in_decoder': 0.5, 
#              'out_decoder': 0.5,
#              'predictor': 0.5}
# weight_decays = {'in_encoder': 0.0001,
#                 'in_decoder': 0.0001,
#              'out_decoder': 0.0001,
#              'predictor': 0.0001}
# num_epochs = 50
# eval_every = 3
# patience = 10000
# num_tries = 1

# trainer = Trainer(jepa=jepa, 
#                   train_dataloader=train_dataloader, 
#                   val_dataloader=val_dataloader, 
#                   initial_lrs=initial_lrs,
#                   lr_decay_gammas=lr_decay_gammas,
#                   lr_decay_periods=lr_decay_periods,
#                   weight_decays=weight_decays)
# jepa = trainer.train(num_epochs, eval_every, patience, num_tries)
