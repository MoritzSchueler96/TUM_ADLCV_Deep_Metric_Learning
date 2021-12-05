import os
import os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn


def make_dir(dir):
    if not osp.isdir(dir):
        os.makedirs(dir)


def set_seeds(seed: int):
    # For reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
