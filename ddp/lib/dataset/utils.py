import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class MixUpCollate:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, self.alpha)
        return mixed_inputs, y_a, y_b, lam
