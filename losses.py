import math
import torch
import torch.nn.functional as F

def nonzero_l1_loss(y_hat, y):
    mask = y != 0
    return F.l1_loss(y_hat[mask], y[mask])

def negative_correlation_loss(y_hat, y):
    """
    Negative correlation loss function for tensors
    
    Returns:
    -1 = perfect positive correlation
    1 = totally negative correlation
    """
    # normalize targets
    y_mean = y.mean(dim=-1).unsqueeze(-1)
    y_std = y.std(dim=-1).unsqueeze(-1)
    y = (y - y_mean) / y_std
    
    # compute correlation
    y_hat_centered = y_hat - y_hat.mean(dim=-1).unsqueeze(dim=-1)
    r = (y * y_hat_centered).sum(dim=-1)
    norms = torch.linalg.norm(y_hat_centered, dim=-1)
    r = (r / norms).mean() / math.sqrt(y.shape[-1])
    return -r

def focal_loss(y_hat, y):
    '''
    Modified focal loss.
    Runs faster and costs a little bit more memory.
    
    Arguments:
        y_hat (batch x N)
        y (batch x N)
    '''
    y = (y != 0).float()
    y_hat = torch.sigmoid(y_hat)
    
    pos_inds = y.eq(1).float()
    neg_inds = y.lt(1).float()

    neg_weights = torch.pow(1 - y, 5)   # change this weight for balance of pos/neg loss
    # clamp min value is set to 1e-12 to maintain the numerical stability
    y_hat = torch.clamp(y_hat, 1e-12)

    pos_loss = torch.log(y_hat) * torch.pow(1 - y_hat, 2) * pos_inds
    neg_loss = torch.log(1 - y_hat) * torch.pow(y_hat, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def bce_loss(y_hat, y):
    y = (y != 0).float()
    y_hat = torch.sigmoid(y_hat)
    return F.binary_cross_entropy(y_hat, y)
