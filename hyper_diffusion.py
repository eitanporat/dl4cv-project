import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from config import FLAGS

class OptimizerBasedDiffusion(nn.Module):
    def __init__(self, time_steps):
        super(OptimizerBasedDiffusion, self).__init__()
        self.sampler = Sampler(time_steps=time_steps)
        self.time_steps = time_steps
        self.ts = torch.zeros(time_steps, dtype=torch.float32)

    def step(self, model, x_t, time_step):
        noise = torch.randn_like(x_t).to(x_t.device)
        ones = torch.ones((x_t.shape[0], ), dtype=torch.float32).to(x_t.device)
        t = ones * (FLAGS.T - 1) * torch.cumsum(F.softmax(self.ts, dim=0), dim=0)[time_step]
        eps = model(x_t, t.int()).to(x_t.device)
        z = torch.cat([x_t, eps, noise], dim=1)
        return self.sampler(time_step, z)

    def forward(self, model, x_T):
        x_t = x_T
        if self.training:
            x_t.requires_grad = True
        for time_step in reversed(range(self.time_steps)):
            if self.training:
                x_t = checkpoint.checkpoint(self.step, model, x_t, time_step)
            else:
                x_t = self.step(model, x_t, time_step)
        return x_t

class SamplerTimestepLayer(nn.Module):
    def __init__(self) -> None:
        super(SamplerTimestepLayer, self).__init__()
        self.a = nn.Parameter(torch.tensor(.9))
        self.b = nn.Parameter(torch.tensor(-.1))
        self.c = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        out = self.a * x[:, :3] + self.b * x[:, 3:6] + self.c * x[:, 6:]
        return out

class Sampler(nn.Module):
    def __init__(self, time_steps=20):
        super(Sampler, self).__init__()
        self.layers = nn.Sequential(*[SamplerTimestepLayer() for _ in range(time_steps)])

    def forward(self, time_step, z_t):
        return self.layers[time_step](z_t)
