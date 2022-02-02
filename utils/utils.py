import os
import os.path as osp
import numpy as np
import random
import torch


def make_dir(dir):
    if not osp.isdir(dir):
        os.makedirs(dir)


def set_seeds(seed: int):
    # For reproducibility
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = f"{seed}"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
