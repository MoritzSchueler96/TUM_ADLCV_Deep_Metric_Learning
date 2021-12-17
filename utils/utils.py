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
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
