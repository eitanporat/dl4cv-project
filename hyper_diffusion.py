import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from config import FLAGS

class LinearSamplerTimestepLayer(nn.Module):
    def __init__(self) -> None:
        super(LinearSamplerTimestepLayer, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.))
        self.b = nn.Parameter(torch.tensor(0.))
        self.c = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        # a * image + b * noise + c * gaussian
        out = self.a * x[:, :3] + self.b * x[:, 3:6] + self.c * x[:, 6:]
        return out

class LinearSampler(nn.Module):
    def __init__(self, time_steps=20):
        super(LinearSampler, self).__init__()
        self.time_steps = time_steps
        self.layers = nn.Sequential(*[LinearSamplerTimestepLayer() for _ in range(time_steps)])

    def step(self, model, x_t, time_step):
        ones = torch.ones((x_t.shape[0], ), dtype=torch.long).to(x_t.device)
        t = ones * int((FLAGS.T - 1) * (time_step) / (self.time_steps - 1))
        noise = torch.randn_like(x_t).to(x_t.device)
        eps = model(x_t, t).to(x_t.device)
        z = torch.cat([x_t, eps, noise], dim=1)
        return self.layers[time_step](z)

    def forward(self, model, x_T):
        x_t = x_T
        if self.training:
            x_t.requires_grad = True
        for time_step in reversed(range(self.time_steps)):
            # time steps
            if self.training:
                x_t = checkpoint.checkpoint(self.step, model, x_t, time_step)
            else:
                x_t = self.step(model, x_t, time_step)
        return x_t

class MomentumSampler(nn.Module):
    def __init__(self, time_steps=10):
        super(MomentumSampler, self).__init__()
        self.time_steps = time_steps
        self.layers = nn.Sequential(*[MomentumSamplerTimeStepLayer(i + 1) for i in reversed(range(time_steps))])

    def step(self, model, x_t, time_step, epsilons):
        ones = torch.ones((x_t.shape[0], ), dtype=torch.long).to(x_t.device)
        t = ones * int((FLAGS.T - 1) * (time_step) / (self.time_steps - 1))
        noise = torch.randn_like(x_t).to(x_t.device)
        eps = model(x_t, t).to(x_t.device)
        epsilons[time_step] = eps
        return self.layers[time_step](x_t, torch.cat(epsilons[time_step:], dim=1), noise)

    def forward(self, model, x_T):
        epsilons = [None] * self.time_steps

        x_t = x_T
        if self.training:
            x_t.requires_grad = True
        
        for time_step in reversed(range(self.time_steps)):
            # time steps
            if self.training:
                x_t = checkpoint.checkpoint(self.step, model, x_t, time_step, epsilons)
            else:
                x_t = self.step(model, x_t, time_step, epsilons)
        
        return x_t

class MomentumSamplerTimeStepLayer(nn.Module):
    def __init__(self, time_steps) -> None:
        super(MomentumSamplerTimeStepLayer, self).__init__()
        self.a = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.zeros(time_steps, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(0.))

    def forward(self, x, epsilons, noise):
        sum_r = torch.einsum('i,bijk->bjk', self.b, epsilons[:, ::3, :, :])
        sum_g = torch.einsum('i,bijk->bjk', self.b, epsilons[:, 1::3, :, :])
        sum_b = torch.einsum('i,bijk->bjk', self.b, epsilons[:, 2::3, :, :])
        epsilons_out = torch.stack([sum_r, sum_g, sum_b], dim=1)
        out = self.a * x + epsilons_out + self.c * noise
        return out
