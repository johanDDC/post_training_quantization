from random import random
from copy import deepcopy
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)


def mixup_data(batch, targets, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = batch.size()[0]
    index = torch.randperm(batch_size)
    index.to(device)

    mixed_x = lam * batch + (1 - lam) * batch[index, :]
    y_a, y_b = targets, targets[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Timer:
    def __init__(self, title=None):
        torch.set_num_threads(1)
        self.title=title
        self.tm = 0

    def __enter__(self):
        self.start = perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_num_threads(8)
        self.tm = perf_counter() - self.start
        if self.title is not None:
            print(f"{self.title} took {round(self.tm, 5)}s.")


