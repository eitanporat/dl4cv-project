from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    if world_size > 1:
        distributed_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                                num_workers=num_workers, sampler=distributed_sampler, drop_last=True)
        return dataloader, distributed_sampler
    
    return DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=True, drop_last=True)