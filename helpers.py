
import os
import time

import torch
from tqdm import trange

from config import FLAGS
from torchvision.utils import make_grid, save_image

from score.both import get_inception_and_fid_score


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

def evaluate(sample_fn, save_images=False):
    if save_images:
        file_dir = f'./generated/{time.strftime("%Y%m%d-%H%M%S")}-{FLAGS.optimizer_kernel_size}-{FLAGS.optimizer_out_channels}-{FLAGS.optimizer_time_steps}'
        os.mkdir(file_dir)

    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sample_fn(x_T.cuda()).detach().cpu()
            images.append((batch_images + 1) / 2)

            if save_images:
                if i % FLAGS.save_every == 0:
                    # convert the tensor of images `images` to a grid of images
                    batch = images[-1]
                    batch = (batch + 1) / 2
                    batch = batch.clamp(0, 1)
                    batch = batch.view(-1, 3, FLAGS.img_size, FLAGS.img_size)
                    # convert to a grid of images
                    grid = make_grid(batch[:64], nrow=8, normalize=True)
                    # save the grid to a file
                    save_image(grid, f'{file_dir}/{i}_eval.png')
    
        images = torch.cat(images, dim=0).numpy()

    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)

    return (IS, IS_std), FID