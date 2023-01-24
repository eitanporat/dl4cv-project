import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizerBasedDiffusion(nn.Module):
    def __init__(self, model, kernel_size, out_channels, time_steps):
        self.model = model
        self.sampler = Sampler(time_steps=time_steps, kernel_size=kernel_size, out_channels=out_channels)
        self.time_steps = time_steps

    def forward(self, x_T):
        z_t = torch.cat([x_T, self.model(x_T, t), torch.randn_like(x_T)], dim=1)
        
        for time_step in reversed(range(self.time_steps)):
            
            # create time steps
            t = torch.ones_like(x_T, dtype=torch.long) * time_step
            
            # call model on first input
            eps = self.model(z_t[:, 0], t)

            # create noise
            noise = torch.randn_like(x_T, requires_grad=False)

            z_t = torch.cat([z_t[:,0], eps, noise], dim=1)
            z_t = self.sampler(time_step, z_t)

        x_0 = z_t[:, 0]
        return torch.clip(x_0, -1, 1)

class SamplerTimestepLayer(nn.Module):
    def __init__(self, kernel_size, out_channels) -> None:
        super(SamplerTimestepLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=3, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(3)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out += residual
        return out

class Sampler(nn.Module):
    def __init__(self, time_steps=20, kernel_size=1, out_channels=3):
        self.layers = nn.ModuleList([SamplerTimestepLayer(kernel_size, out_channels) for _ in range(time_steps)])

    def forward(self, time_step, z_t):
        return self.layers[time_step](z_t)
