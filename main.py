import copy
import json
import os
import warnings
from config import FLAGS
import torch
# from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
import time
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model, save_images=False):
    model.eval()
    if save_images:
        file_dir = f'./generated/{time.strftime("%Y%m%d-%H%M%S")}-{FLAGS.optimizer_kernel_size}-{FLAGS.optimizer_out_channels}-{FLAGS.optimizer_time_steps}'
        os.mkdir(file_dir)

    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).detach().cpu()
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
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)

    return (IS, IS_std), FID, images


def train():
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model.load_state_dict(ckpt['net_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=16)


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
