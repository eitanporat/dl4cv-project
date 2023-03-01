import os
from absl import app
import torch
from torchvision.utils import save_image, make_grid
from model import UNet
from config import FLAGS
from helpers import evaluate

from ddim import GaussianDiffusionTimestepsSampler
from hyper_diffusion import MomentumSampler

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return {k.replace('.module', ''): v for k, v in checkpoint.items()}

def eval(argv):
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)

    if FLAGS.model_checkpoint:
        ckpt = torch.load(FLAGS.model_checkpoint)
        ckpt = ckpt['net_model'] if 'net_model' in ckpt else ckpt
        model.load_state_dict(ckpt)
        model = model.cuda()
        model.eval()

    if FLAGS.sampler_type == 'ddim':
        sampler = GaussianDiffusionTimestepsSampler(model, 
                    FLAGS.beta_1, 
                    FLAGS.beta_T, 
                    T_orig = FLAGS.T, 
                    T_reduced = FLAGS.T_reduced, 
                    img_size=FLAGS.img_size,
                    mean_type=FLAGS.mean_type, 
                    var_type=FLAGS.var_type).cuda()
        
        sample_fn = lambda x: sampler.ddim_sample(x)

    if FLAGS.sampler_type == 'momentum':
        sampler = MomentumSampler(FLAGS.optimizer_time_steps).cuda()

        if FLAGS.sampler_checkpoint:
            print('Loading sampler checkpoint...')
            checkpoint = load_checkpoint(FLAGS.sampler_checkpoint)
            sampler.load_state_dict(checkpoint)

        if FLAGS.time_embedding_checkpoint:
            print('Loading checkpoint...')
            checkpoint = load_checkpoint(FLAGS.time_embedding_checkpoint)
            model.time_embedding.load_state_dict(checkpoint)

        sample_fn = lambda x: sampler(model, x)

    sampler.eval()
    print(evaluate(sample_fn, save_images = True))


if __name__ == '__main__':
    app.run(eval)
