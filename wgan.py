from helpers import infiniteloop
from config import FLAGS
from wgan_utils import to_np, train, freeze
from torchvision import utils
from itertools import chain
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class WGAN_GP(object):
    def __init__(self, diffusion_model, discriminator, sampler, device):
        print("WGAN_GradientPenalty init model.")
        self.diffusion_model = diffusion_model
        self.D = discriminator
        self.G = sampler
        self.C = 3
        self.device = device

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = FLAGS.batch_size

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(
            self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(
            chain(
                self.G.parameters(),
                self.diffusion_model.module.time_embedding.parameters()),
            lr=self.learning_rate, betas=(self.b1, self.b2))

        # self.logger = Logger('./logs')
        # self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = FLAGS.generator_iters
        self.critic_iter = FLAGS.critic_iters
        self.lambda_term = 10

    def train(self, train_loader):
        self.t_begin = t.time()
        self.data = infiniteloop(train_loader)

        for g_iter in range(self.generator_iters):
            for d_iter in range(self.critic_iter):
                wasserstein_distance, d_loss, d_loss_real, d_loss_fake = self.d_step()
                self.log_(
                    f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            g_cost = self.g_step()
            self.log_(
                f'Generator iteration: {g_iter}/{self.generator_iters}, g_cost: {g_cost.item()}')

            if g_iter % FLAGS.save_every == 0 and self.device == 0:
                self.log(wasserstein_distance.item(),
                         d_loss.item(), g_cost.item(),
                         d_loss_real.item(),
                         d_loss_fake.item(),
                         g_iter)

                self.t_end = t.time()
                self.log_(
                    'Time of training-{}'.format((self.t_end - self.t_begin)))
                self.save_model()

    def log_(self, x):
        if self.device == 0:
            print(x)

    def g_step(self):
        train(self.G)
        freeze(self.D)

        z = torch.randn(self.batch_size, 3, 32, 32).to(self.device)
        fake_images = self.G(self.diffusion_model, z)
        g_loss = self.D(fake_images)
        g_loss = g_loss.mean()
        g_cost = -g_loss
        g_cost.backward()

        self.g_optimizer.step()

        return g_cost.data

    def d_step(self):
        train(self.D)
        freeze(self.G)

        images = next(self.data).to(self.device)

        # Check for batch to have full batch_size
        if (images.size()[0] != self.batch_size):
            return

        d_loss_real = self.D(images)
        d_loss_real = d_loss_real.mean()

        z = torch.randn(self.batch_size, 3, 32, 32).to(self.device)

        fake_images = self.G(self.diffusion_model, z)
        d_loss_fake = self.D(fake_images)
        d_loss_fake = d_loss_fake.mean()

        gradient_penalty = self.calculate_gradient_penalty(images, fake_images)

        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        d_loss.backward()

        wasserstein_distance = d_loss_real - d_loss_fake
        self.d_optimizer.step()

        return wasserstein_distance.data, d_loss.data, d_loss_real.data, d_loss_fake.data

    def log(self, wasserstein_distance, d_loss, g_cost, d_loss_real, d_loss_fake, g_iter):
        self.save_model()

        print(f'Iteration {g_iter}\n{wasserstein_distance=:.3f}\n{d_loss=:.3f}\n{g_cost=:.3f}\n{d_loss_real=:.3f}\n{d_loss_fake=:.3f}')
        if not os.path.exists('training_result_images/'):
            os.makedirs('training_result_images/')

        # Denormalize images and save them in grid 8x8
        z = torch.randn(64, 3, 32, 32).to(self.device)
        samples = self.G(self.diffusion_model, z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()[:64]
        grid = utils.make_grid(samples)
        utils.save_image(
            grid, 'training_result_images/img_generator_iter_{}.png'.format(str(g_iter).zfill(3)))

        time = t.time() - self.t_begin
        self.log_("Generator iter: {}".format(g_iter))
        self.log_("Time {}".format(time))

        # info = {'Wasserstein Distance': wasserstein_distance, 'Loss D': d_loss,
        #         'Loss G': g_cost, 'Loss D Real': d_loss_real, 'Loss D Fake': d_loss_fake}

        # for tag, value in info.items():
        #     self.logger.scalar_summary(tag, value.cpu(), g_iter + 1)

        # info = {'real_images': self.real_images(images, self.number_of_images),
        #         'generated_images': self.generate_img(z, self.number_of_images)}

        # for tag, images in info.items():
        #     self.logger.image_summary(tag, images, g_iter + 1)

    def evaluate(self, test_loader, D_model_path, G_model_path, time_embedding_path, diffusion_model_path):
        self.load_model(D_model_path, G_model_path,
                        time_embedding_path, diffusion_model_path)
        z = torch.randn(self.batch_size, 100, 1, 1).to(self.device)
        samples = self.G(self.diffusion_model, z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))\
            .to(self.device)
        interpolated = eta * real_images + \
            ((1 - eta) * fake_images).to(self.device)

        interpolated = Variable(interpolated, requires_grad=True)
        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)
        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(self.diffusion_model, z).data.cpu().numpy()[
            :number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        torch.save(
            self.diffusion_model.module.time_embedding.state_dict(), './time_embedding.pkl')

        print('Models save to ./generator.pkl & ./discriminator.pkl & time_embedding.pkl')

    def load_model(self, D_model_filename, G_model_filename, diffusion_model_filename, time_embedding_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)

        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))

        self.diffusion_model.load_state_dict(
            torch.load(diffusion_model_filename)['net_model'])
        self.diffusion_model.time_embedding.load_state_dict(
            torch.load(time_embedding_filename))

        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1).to(self.device)
        z1 = torch.randn(1, 100, 1, 1).to(self.device)
        z2 = torch.randn(1, 100, 1, 1).to(self.device)

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(self.diffusion_model, z_intp)
            fake_im = fake_im.mul(0.5).add(0.5)  # denormalize
            images.append(fake_im.view(self.C, 32, 32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int)
        utils.save_image(
            grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
