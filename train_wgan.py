from absl import app
import torch
from discriminator import Discriminator
from hyper_diffusion import MomentumSampler
from model import UNet
from config import FLAGS
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import sys
from cifar10 import prepare
from distributed_utils import setup, cleanup
from wgan import WGAN_GP

def train(rank, world_size):
    if world_size > 1:
        setup(rank, world_size)

    FLAGS(argv=sys.argv)
    
    diffusion_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(rank)
    
    discriminator = Discriminator(channels=3).to(rank)    
    sampler = MomentumSampler(FLAGS.optimizer_time_steps).to(rank)

    # load model and evaluate
    if FLAGS.model_checkpoint:
        checkpoint = torch.load(FLAGS.model_checkpoint)
        diffusion_model.load_state_dict(checkpoint['net_model'])

    if FLAGS.time_embedding_checkpoint:
        checkpoint = torch.load(FLAGS.time_embedding_checkpoint)
        diffusion_model.time_embedding.load_state_dict(checkpoint)

    if FLAGS.discriminator_checkpoint:
        checkpoint = torch.load(FLAGS.discriminator_checkpoint)
        discriminator.load_state_dict(checkpoint)

    if FLAGS.sampler_checkpoint:
        checkpoint = torch.load(FLAGS.sampler_checkpoint)
        sampler.load_state_dict(checkpoint)

    if FLAGS.parallel:
        diffusion_model = DDP(diffusion_model, device_ids=[rank])
        sampler = DDP(sampler, device_ids=[rank])
        discriminator = DDP(discriminator, device_ids=[rank])

    diffusion_model.train()

    wgan = WGAN_GP(diffusion_model, discriminator, sampler, rank)

    dataloader, _ = prepare(
        rank, world_size, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers)

    torch.manual_seed(1337 + rank)
    wgan.train(dataloader)

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
