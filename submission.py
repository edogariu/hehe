import pandas as pd
import numpy as np
import tqdm
import torch

from utils import device
from model import ModelWrapper
from datasets import SubmissionDataset
from ensemble_cite import CiteInferencer

def generate_submissions(cite_model: ModelWrapper, multi_model: ModelWrapper):    
    print('reading quickstart csv')
    SPLIT = 6812820  # index of the first multi row (index after last cite row)
    submission = pd.read_csv('submissions/quickstart.csv', index_col='row_id')
    
    # get cite submissions first
    print('inferencing on cite model')
    cite_dataset = SubmissionDataset('cite', 'naive')
    cite_loader = cite_dataset.get_dataloader(1)
    preds = np.zeros((len(cite_dataset), 140), dtype=np.float32)
    # cite_model.eval().to(device)
    with torch.no_grad():
        i = 0
        for x, cell_id in tqdm.tqdm(cite_loader):
            batch_size = x.shape[0]
            x = x.to(device)
            out = cite_model.infer(x, cell_id)
            preds[i:i + batch_size] = out.cpu().detach().numpy()
            i += batch_size
    
    # data leak type beat for the first 7476 cite targets
    train_targets = pd.read_hdf('data/train_cite_targets.h5', start=0, stop=8000)
    preds[:7476] = train_targets.values[:7476]
    
    cite_submission = preds.ravel()
    assert len(cite_submission) == SPLIT
    submission[:SPLIT] = np.expand_dims(cite_submission, axis=1)
    
    # get multi submissions next
    print('reading eval ids')
    eval_ids = pd.read_csv('data/evaluation_ids.csv')
    multi_loader = None
    
    print('inferencing on multi model')
    multi_dataset = SubmissionDataset('multi')
    multi_loader = multi_dataset.get_dataloader(128)
    preds = np.zeros((len(multi_dataset), 23418), dtype=np.float32)
    multi_model.eval().to(device)
    with torch.no_grad():
        i = 0
        for x, day in tqdm.tqdm(multi_loader):
            batch_size = x.shape[0]
            x, day = x.to(device), day.to(device)
            out = multi_model.infer(x)
            preds[i:i + batch_size] = out.cpu().detach().numpy()
            i += batch_size
    
    preds = multi_model.infer_on_whole_dataset(multi_dataset, 256)
    print(len(multi_dataset), preds.shape)
    multi_submission = preds.ravel()
    assert len(multi_submission) == len(submission) - SPLIT
    submission[SPLIT:] = multi_submission
    
    assert not submission.isna().any()
    return submission

if __name__ == '__main__':
    # cite_model = Model(RNA2Protein(22050, 140, 512, 5, 2, 3, 'linear'), 'rna_to_protein_linear')
    # cite_model.load_checkpoint()

    # multi_model = Model(DNA2RNA(20000, 23418, 64, 2, 2, 'attention'), 'dna_to_rna_nc')
    # multi_model.load_checkpoint()
    from silly import SillyInference
    # from cite import CiteModel
    # cite_model = CiteModel(4110, 140, 8)
    # cite_model.load_state_dict(torch.load('checkpoints/models/cite_coding_pca copy.pth'))
    
    cite_model = CiteInferencer()
    multi_model = SillyInference()
        
    submission = generate_submissions(cite_model, multi_model)
    print('writing submission to csv')
    submission.to_csv('submissions/submission.csv')
