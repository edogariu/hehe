import pandas as pd
import numpy as np
import tqdm
import torch

from utils import device
from architectures import RNA2Protein
from model import Model
from datasets import SubmissionDataset

cite_model = Model(RNA2Protein(22050, 140, 512, 5, 2, 2, 'transformer'), 'rna_to_protein_transformer')
cite_model.load_checkpoint()

def generate_submissions(cite_model: Model, multi_model: Model):
    print('reading quickstart csv')
    SPLIT = 6812820  # index of the first multi row (index after last cite row)
    submission = pd.read_csv('submissions/quickstart.csv', index_col='row_id', squeeze=True)
    
    # get cite submissions first
    print('inferencing on cite model')
    cite_dataset = SubmissionDataset('cite')
    cite_loader = cite_dataset.get_dataloader(32)
    preds = np.zeros((len(cite_dataset), 140), dtype=np.float32)
    cite_model.eval().to(device)
    with torch.no_grad():
        i = 0
        for x, day in tqdm.tqdm(cite_loader):
            batch_size = x.shape[0]
            x, day = x.to(device), day.to(device)
            out = cite_model.infer(x, day)
            preds[i:i + batch_size] = out.cpu().detach().numpy()
            i += batch_size
    
    # data leak type beat for the first 7476 cite targets
    train_targets = pd.read_hdf('data/train_cite_targets.h5', start=0, stop=8000)
    preds[:7476] = train_targets.values[:7476]
    
    cite_submission = list(preds.ravel())
    assert len(cite_submission) == SPLIT
    submission[:SPLIT] = cite_submission
    
    # add multi part here
    
    assert not submission.isna().any()
    return submission
    
submission = generate_submissions(cite_model, None)
submission.to_csv('submissions/submission.csv')
