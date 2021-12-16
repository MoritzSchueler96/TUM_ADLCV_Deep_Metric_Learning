import os
import os.path as osp
import random
import numpy as np
import torch


def make_dir(dir):
    if not osp.isdir(dir):
        os.makedirs(dir)

def set_seeds(seed: int):
    # For reproducibility
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
