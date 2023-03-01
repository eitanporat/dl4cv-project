from hyper_diffusion import MomentumSampler
import os
from absl import app
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from model import UNet
from score.kid import KernelInceptionDistance
from config import FLAGS
from tqdm import tqdm
import warnings
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
from itertools import chain
from helpers import infiniteloop
from lion import Lion

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
    else:
        return DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=True, drop_last=True)
    return dataloader, distributed_sampler


def setup(rank, world_size):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1232'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def monitor_loss(sampler, lr):
    for i, layer in enumerate(sampler.module.layers):
        grad = layer.a.grad
        weight = layer.a
        print(f'Layer {i}:\tA={layer.a.item()}\tgrad/weight ratio: {lr * grad / weight:.6f}')
        grad = layer.b.grad.std()
        weight = layer.b.std()
        print(f'Layer {i}:\tB={layer.b.tolist()}\tgrad/weight ratio: {lr * grad / weight:.6f}')

        grad = layer.c.grad
        weight = layer.c
        print(f'Layer {i}:\tC\tgrad/weight ratio: {lr * grad / weight:.6f}')


def train(rank, world_size):
    if world_size > 1:
        setup(rank, world_size)

    FLAGS(argv=sys.argv)

    progress_bar = tqdm(range(0, 10000000000, world_size)
                        ) if rank == 0 else range(10000000000)

    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    if FLAGS.model_checkpoint:
        checkpoint = torch.load(FLAGS.model_checkpoint)
        model.load_state_dict(checkpoint['net_model'])

    if FLAGS.time_embedding_checkpoint:
        checkpoint = torch.load(FLAGS.time_embedding_checkpoint)
        model.time_embedding.load_state_dict(checkpoint)

    model = model.to(rank)
    model.train()

    sampler = MomentumSampler(FLAGS.optimizer_time_steps).to(rank)

    if FLAGS.sampler_checkpoint:
        checkpoint = torch.load(FLAGS.sampler_checkpoint)
        sampler.load_state_dict(checkpoint)

    # if checkpoint flag is set, load checkpoint
    kid = KernelInceptionDistance(parallel=FLAGS.parallel).to(rank)

    dataloader, distributed_sampler = prepare(
        rank, world_size, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers)
    dataloader_iter = infiniteloop(dataloader)

    if FLAGS.parallel:
        sampler = DDP(sampler, device_ids=[rank])
        model = DDP(model, device_ids=[rank])

    # model.time_embedding.parameters()
    if FLAGS.optimizer_type == 'adamw':
        optimizer_class = torch.optim.AdamW
    elif FLAGS.optimizer_type == 'adam':
        optimizer_class = torch.optim.Adam
    elif FLAGS.optimizer_type == 'sgd':
        optimizer_class = torch.optim.SGD
    elif FLAGS.optimizer_type == 'lion':
        optimizer_class = Lion

    if FLAGS.train_time_embedding:
        optim = optimizer_class(chain(model.module.time_embedding.parameters(), sampler.parameters()), lr=FLAGS.lr)
    else:
        optim = optimizer_class(sampler.parameters(), lr=FLAGS.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[500, 1500], gamma=0.1)

    if rank == 0:
        if FLAGS.file_dir:
            file_dir = FLAGS.file_dir + f'/{time.strftime("%Y%m%d-%H%M%S")}'
        else:
            file_dir = f'./generated/{time.strftime("%Y%m%d-%H%M%S")}'
        os.mkdir(file_dir)

        with open(f'{file_dir}/config.txt', 'w+') as config_file:
            # write flag to config file
            for key, value in FLAGS.flag_values_dict().items():
                config_file.write(f'{key}={value}\n')

    torch.manual_seed(1337 + rank)

    for epoch in progress_bar:
        distributed_sampler.set_epoch(epoch)
        
        optim.zero_grad()
        x_T = torch.randn((FLAGS.batch_size, 3, FLAGS.img_size, FLAGS.img_size)).to(rank)
        real_images = next(dataloader_iter).to(rank)
        real_images = 2 * real_images - 1
        images = sampler(model, x_T)
        loss = kid(images, real_images)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sampler.parameters(), 1)
        optim.step()

        if rank == 0:
            progress_bar.set_postfix(loss=loss.item())

        if rank == 0 and epoch % FLAGS.save_every == 0:
            # convert the tensor of images `images` to a grid of images
            images = images.detach().cpu()
            images = (images + 1) / 2
            images = images.clamp(0, 1)
            images = images.view(-1, 3, FLAGS.img_size, FLAGS.img_size)
            # convert to a grid of images
            grid = make_grid(images[:64], nrow=8, normalize=True)
            # save the grid to a file
            save_image(grid, f'{file_dir}/{epoch}-{loss.item():.2f}.png')
            monitor_loss(sampler, optim.param_groups[0]['lr'])

        if rank == 0 and epoch % 100 == 0 and epoch != 0:
            torch.save(sampler.module.state_dict(),
                       f'{file_dir}/{loss.item():.4f}-sampler.ckpt')
            # torch.save(model.time_embedding.state_dict(),
            #            f'{file_dir}/time-embedding-{loss.item():.4f}.ckpt')
            if FLAGS.train_time_embedding:
                torch.save(model.module.time_embedding.state_dict(), f'{file_dir}/{loss.item():.4f}-time-embedding.ckpt')


    if rank == 0:
        torch.save(sampler.module.state_dict(), f'{file_dir}/sampler-last.ckpt')
        torch.save(model.module.state_dict(), f'{file_dir}/model-last.ckpt')

    if FLAGS.parallel:
        cleanup()


def main(argv):
    if FLAGS.parallel:
        torch.multiprocessing.set_start_method('spawn', force=True)

    # suppress annoying inception_v3 initialization warning
    print('Training...')

    if FLAGS.parallel:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

        world_size = n_gpus

        processes = []
        for rank in range(world_size):
            p = mp.Process(target=train, args=(rank, world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        train(0, 1)

if __name__ == '__main__':
    app.run(main)
