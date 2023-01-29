from hyper_diffusion import OptimizerBasedDiffusion
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
from main import evaluate, infiniteloop


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
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def monitor_loss(sampler, lr):
    for i, layer in enumerate(sampler.module.sampler.layers):
        grad = layer.a.grad
        weight = layer.a
        print(
            f'Layer {i}:\tA={layer.a.item()}\tgrad/weight ratio: {lr * grad / weight:.6f}')
        grad = layer.b.grad
        weight = layer.b
        print(
            f'Layer {i}:\tB={layer.b.item()}\tgrad/weight ratio: {lr * grad / weight:.6f}')
        # grad = layer.c.grad.std()
        # weight = layer.c.std()
        # print(f'Layer {i}:\tC\tgrad/weight ratio: {lr * grad / weight:.6f}')


def train(rank, world_size):
    if world_size > 1:
        setup(rank, world_size)

    FLAGS(argv=sys.argv)

    progress_bar = tqdm(range(0, 100_000, world_size)
                        ) if rank == 0 else range(100_000)

    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['net_model'])

    model = model.to(rank)
    model.train()

    sampler = OptimizerBasedDiffusion(FLAGS.optimizer_time_steps).to(rank)

    # if checkpoint flag is set, load checkpoint
    kid = KernelInceptionDistance(parallel=FLAGS.parallel).to(rank)

    dataloader, distributed_sampler = prepare(
        rank, world_size, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers)
    dataloader_iter = infiniteloop(dataloader)

    if FLAGS.parallel:
        sampler = DDP(sampler, device_ids=[rank])

    if FLAGS.checkpoint:
        print('Loading checkpoint...')
        checkpoint = torch.load(FLAGS.checkpoint)
        sampler.module.load_state_dict(checkpoint)

    optim = torch.optim.Adam(sampler.parameters(), lr=FLAGS.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 500], gamma=0.1)

    if rank == 0:
        file_dir = f'./generated/{time.strftime("%Y%m%d-%H%M%S")}-{FLAGS.optimizer_kernel_size}-{FLAGS.optimizer_out_channels}-{FLAGS.optimizer_time_steps}'
        os.mkdir(file_dir)

    torch.manual_seed(1337 + rank)

    for epoch in progress_bar:
        distributed_sampler.set_epoch(epoch)
        x_T = torch.randn((FLAGS.batch_size, 3, FLAGS.img_size, FLAGS.img_size)).to(rank)
        real_images = next(dataloader_iter).to(rank)
        real_images = 2 * real_images - 1
        optim.zero_grad()
        images = sampler(model, x_T)
        loss = kid(images, real_images)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sampler.parameters(), 1)
        optim.step()
        scheduler.step()

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
                       f'{file_dir}/sampler-{loss.item():.4f}.ckpt')

    if rank == 0:
        torch.save(sampler.module.state_dict(), f'{file_dir}/last.ckpt')

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


def eval(argv):
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['net_model'])

    model = model.cuda()
    model.eval()

    sampler = OptimizerBasedDiffusion(FLAGS.optimizer_time_steps).cuda()

    # if checkpoint flag is set, load checkpoint
    if FLAGS.checkpoint:
        print('Loading checkpoint...')
        checkpoint = torch.load(FLAGS.checkpoint)
        sampler.load_state_dict(
            {k.replace('.module', ''): v for k, v in checkpoint.items()})

    print(evaluate(sampler, model))


if __name__ == '__main__':
    app.run(main)
