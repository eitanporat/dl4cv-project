import warnings
import os
import torch.distributed as dist

def setup(rank, world_size):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1232'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()