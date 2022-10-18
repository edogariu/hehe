import torch

import architectures
from datasets import H5Dataset, SparseDataset, NaiveDataset
from trainer import Trainer
from model import Model
from utils import count_parameters

def train():
    # # ----------------------------------------------------------------------------------------------------
    # # ------------------------------------   Multiome Training   -----------------------------------------
    # # ----------------------------------------------------------------------------------------------------

    # # ------------------------------------- hyperparameters -------------------------------------------------
    # batch_size = 16
    # model_name = 'dna_to_rna_sigmoid'

    # initial_lr = 0.04
    # lr_decay_period = 5
    # lr_decay_gamma = 0.5
    # weight_decay = 0.0001

    # num_epochs = 20
    # eval_every = 1
    # patience = 4
    # num_tries = 4

    # model = architectures.DNA2RNA()
    # # --------------------------------------------------------------------------------------------------------

    # print('preparing datasets!')
    # train_dataloader = H5Dataset('train', 'multi').get_dataloader(batch_size)
    # val_dataloader = H5Dataset('val', 'multi').get_dataloader(batch_size)
    # test_dataloader = H5Dataset('test', 'multi').get_dataloader(1)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # # model.load_state_dict(torch.load('checkpoints/models/{}.pth'.format(model_name)))
    # trainer = Trainer(Model(model, model_name), train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
    # trainer.train(num_epochs, eval_every, patience, num_tries)

    # # ----------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------





    # ----------------------------------------------------------------------------------------------------
    # ------------------------------------   CITEseq Training   ------------------------------------------
    # ----------------------------------------------------------------------------------------------------

    # ------------------------------------- hyperparameters -------------------------------------------------
    batch_size = 128
    model_name = 'rna_to_protein_enformer'

    initial_lr = 0.025
    lr_decay_period = 20
    lr_decay_gamma = 0.75
    weight_decay = 0.0001

    num_epochs = 100
    eval_every = 3
    patience = 3
    num_tries = 15

    # model = architectures.RNA2Protein(in_dim=22050, out_dim=140, hidden_dim=140, 
    #                                   coding_head_length=2, other_head_length=2, body_length=3, 
    #                                   use_pretrained=True, freeze_heads=True)
    # model = architectures.Encoder(in_dim=22050, out_dim=140, num_channels=256, tower_length=6, body_length=11, body_type='enformer', pooling_type='attention', output_2d=False)
    model = architectures.Test()
    # --------------------------------------------------------------------------------------------------------

    print('preparing datasets!')
    train_dataloader = NaiveDataset('train', 'cite').get_dataloader(batch_size, num_workers=4)
    val_dataloader = NaiveDataset('val', 'cite').get_dataloader(batch_size, num_workers=4)
    # train_dataloader = SparseDataset('train', 'cite').get_dataloader(batch_size)
    # val_dataloader = SparseDataset('val', 'cite').get_dataloader(batch_size)
    test_dataloader = H5Dataset('test', 'cite').get_dataloader(1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model.load_state_dict(torch.load('checkpoints/models/{}.pth'.format(model_name)))
    trainer = Trainer(Model(model, model_name), train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay)
    trainer.train(num_epochs, eval_every, patience, num_tries)

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

if __name__ == '__main__':
    train()