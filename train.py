from hyper_diffusion import OptimizerBasedDiffusion
import os
from absl import app
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from model import UNet
from score.kid import KernelInceptionDistance
from config import FLAGS
from tqdm import tqdm
import warnings
from torchviz import make_dot, make_dot_from_trace
import time

# TODO: change device
device = 'cuda:0'

def train():
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout).to(device)
    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['net_model'])

    # for param in model.parameters():
    #     param.requires_grad = False
    # model.eval()

    sampler = OptimizerBasedDiffusion(FLAGS.optimizer_kernel_size, FLAGS.optimizer_out_channels, FLAGS.optimizer_time_steps).to(device)
    kid = KernelInceptionDistance(parallel=FLAGS.parallel).to(device)
    
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    dataloader = DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    
    dataloader_iter = iter(dataloader)

    optim = torch.optim.Adam(sampler.parameters(), lr=0.01)
    progress_bar = tqdm(range(FLAGS.total_steps))

    file_dir = f'./generated/{time.strftime("%Y%m%d-%H%M%S")}-{FLAGS.optimizer_kernel_size}-{FLAGS.optimizer_out_channels}-{FLAGS.optimizer_time_steps}'
    os.mkdir(file_dir)

    torch.manual_seed(1337)
    x_T = torch.randn((FLAGS.batch_size, 3, FLAGS.img_size, FLAGS.img_size), device=device)
    to_pil = transforms.ToPILImage()
    
    for i in progress_bar:
        real_images = next(dataloader_iter)[0].to(device)
        optim.zero_grad()
        images = sampler(model, x_T.clone())
        # we don't care about labels (for now)
        # real_images = next(dataloader_iter)[0].to(device)
        loss = kid(images, real_images)
        loss.backward()
        if i % 50 == 0:
            v = torch.cat([images[0], images[1], images[2]], dim=2).clip(-1, 1)
            v = (v + 1) / 2
            to_pil(v).save(f'{file_dir}/{i}-{loss.item():.2f}.png')

        # make_dot(loss, params=dict(sampler.named_parameters())).render("attached", format="png")
        optim.step()
        progress_bar.set_postfix(loss=loss.item())


    return sampler

def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print('Training...')
    train()

if __name__ == '__main__':
    app.run(main)
