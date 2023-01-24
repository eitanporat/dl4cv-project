from hyper_diffusion import OptimizerBasedDiffusion
import os

import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

from model import UNet
from score.fid import get_fid_score

# TODO: change device
device = torch.device('cuda:0')

def train(FLAGS):
    # dataset
    # dataset = CIFAR10(
    #     root='./data', train=True, download=True,
    #     transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=FLAGS.batch_size, shuffle=True,
    #     num_workers=FLAGS.num_workers, drop_last=True)

    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    sampler = OptimizerBasedDiffusion(model, FLAGS.optimizer_kernel_size, FLAGS.optimizer_out_channels, FLAGS.time_steps).to(device)
    
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['net_model'])

    for _ in range(FLAGS.epochs):
        x_T = torch.randn((FLAGS.batch_size, 3, FLAGS.img_size, FLAGS.img_size))
        images = sampler(x_T)

        (IS, IS_std), FID = get_fid_score(
            images, FLAGS.fid_cache, num_images=FLAGS.num_images,
            use_torch=FLAGS.fid_use_torch, verbose=True)

        loss = FID.backward()



