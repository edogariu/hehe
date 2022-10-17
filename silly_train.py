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

    for i in range(len(cite_keys)):
        cite_keys[i] = cite_keys[i].split('_')[0]

    # inputs
    multi_idxs_shared = []
    cite_idxs_shared = []

    # targets
    multi_idxs_not_shared = []
    cite_idxs_not_shared = []
    for i, s in enumerate(multi_keys):
        if s in cite_keys:
            multi_idxs_shared.append(i)
        else:
            multi_idxs_not_shared.append(i)
    for i, s in enumerate(cite_keys):
        if s in multi_keys:
            cite_idxs_shared.append(i)
        else:
            cite_idxs_not_shared.append(i)    

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------      HYPERPARAMETERS      ---------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------

    METHOD = 'cite'  # change this to decide which model to train
    model_filename = 'rna_to_protein'

    alpha = 0.00

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

    input_idxs = multi_idxs_shared if METHOD == 'multi' else cite_idxs_shared
    target_idxs = multi_idxs_not_shared if METHOD == 'multi' else cite_idxs_not_shared
    input_idxs = torch.tensor(input_idxs, requires_grad=False).cuda()
    target_idxs = torch.tensor(target_idxs, requires_grad=False).cuda()

    train_dataset = H5Dataset('train', METHOD, days=[2, 3, 4])
    val_dataset = H5Dataset('val', METHOD, days=[2, 3, 4])
    test_dataset = H5Dataset('test', METHOD, days=[2, 3, 4])

    # model = RNAUnifier([len(input_idxs), 6400, 6400, len(target_idxs)], dropout=dropout)
    # model = Encoder(len(input_idxs), len(target_idxs), 256, 8, 6, 'enformer', 'max', output_2d=False)

    # model = Encoder(len(input_idxs), 140, 368, 6, 7, 'enformer', 'attention', output_2d=False)
    model = Encoder(len(input_idxs), 140, 128, 6, 6, 'enformer', 'attention', output_2d=False)
    device = torch.device('cuda')

    train_dataloader = train_dataset.get_dataloader(batch_size)
    val_dataloader = None #val_dataset.get_dataloader(batch_size)

    class PredLoss():
        def __init__(self, model, alpha, input_idxs, target_idxs, method):
            self.model = model
            self.alpha = alpha
            self.input_idxs = input_idxs
            self.target_idxs = target_idxs
            self.method = method
        
        def loss(self, x, day, y):
            rna = x.to(device)[:, self.input_idxs]
            protein_hat = self.model(rna, day)
            loss = F.mse_loss(protein_hat, y)
            return loss

        def error(self, x, day, y):
            rna = x.to(device)[:, self.input_idxs]
            protein_hat = self.model(rna, day)
            loss = F.mse_loss(protein_hat, y)
            return loss.item()

        # def loss(self, x, day, y):
        #     rna = y.to(device) if self.method == 'multi' else x.to(device)
        #     inputs, targets = rna[:, self.input_idxs], rna[:, self.target_idxs]
        #     # sparsity_predictions, regressions = self.model(inputs)
        #     sparsity_predictions = self.model(inputs, day)
            
        #     mask = targets != 0

        #     # loss = F.binary_cross_entropy(sparsity_predictions, mask.float()) + self.alpha * F.mse_loss(regressions[mask], targets[mask])
        #     loss = F.binary_cross_entropy_with_logits(sparsity_predictions, mask.float())
        #     return loss
            
        # def error(self, x, day, y):
        #     rna = y.to(device) if self.method == 'multi' else x.to(device)
        #     inputs, targets = rna[:, self.input_idxs], rna[:, self.target_idxs]
        #     # sparsity_predictions, regressions = self.model(inputs)
        #     sparsity_predictions = self.model(inputs, day)
            
        #     # error = F.binary_cross_entropy(sparsity_predictions, (targets != 0).float())
        #     error = F.binary_cross_entropy_with_logits(sparsity_predictions, (targets != 0).float())
        #     print((targets != 0).float().sum() + (inputs != 0).float().sum())
        #     return error.item()

    l = PredLoss(model, alpha, input_idxs, target_idxs, METHOD)
    # model.load_state_dict(torch.load(os.path.join(TOP_DIR_NAME, 'checkpoints', 'models', '{}.pth'.format(model_filename))))
    trainer = Trainer(model, model_filename, l.loss, l.error, train_dataloader, val_dataloader, initial_lr, lr_decay_period, lr_decay_gamma, weight_decay, grad_accumulation_steps=1)
    trainer.train(num_epochs, eval_every, patience, num_tries)

    print('loading model')
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


# CODE GRAVEYARD

# def sparse_loss(x, num):
#     """
#     L1 norm of `num` smallest values of `x` .
#     Assumes `x` is a `(N, L)` tensor, and loss occurs over the `L` dimension and is then averaged over the `N` dimension.
#     If `L >= num`, returns L1 norm of `x`
#     """
#     n = min(num, x.shape[1])
#     vals, _ = x.sort(dim=-1)
#     vals = vals[:, :n]
#     loss = torch.linalg.norm(vals, ord=1, dim=-1)
#     return loss.mean()

# class PredLoss():
#     def __init__(self, model, num, alpha, input_idxs, target_idxs, method):
#         self.model = model
#         self.num = num
#         self.alpha = alpha
#         self.input_idxs = input_idxs
#         self.target_idxs = target_idxs
#         self.method = method
    
#     def loss(self, x, day, y):
#         rna = y.to(device) if self.method == 'multi' else x.to(device)
#         inputs, targets = rna[:, self.input_idxs], rna[:, self.target_idxs]
#         out = self.model(inputs)
        
#         loss = F.mse_loss(out, targets) + self.alpha * sparse_loss(out, self.num) / self.num
#         return loss
        
#     def error(self, x, day, y):
#         rna = y.to(device) if self.method == 'multi' else x.to(device)
#         inputs, targets = rna[:, self.input_idxs], rna[:, self.target_idxs]
#         out = self.model(inputs)
        
#         error = F.mse_loss(out, targets)
#         print(out.squeeze()[:20])
#         print(targets.squeeze()[:20])
#         print(error.item())
#         print()
#         return error.item()