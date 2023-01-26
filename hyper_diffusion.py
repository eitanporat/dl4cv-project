import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FLAGS

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


class OptimizerBasedDiffusion(nn.Module):
    def __init__(self, kernel_size, out_channels, time_steps):
        super(OptimizerBasedDiffusion, self).__init__()
        self.sampler = Sampler(time_steps=time_steps, kernel_size=kernel_size, out_channels=out_channels)

        self.time_steps = time_steps

    def forward(self, model, x_T):
        z = torch.zeros((x_T.shape[0], x_T.shape[1] * 3, x_T.shape[2], x_T.shape[3]), device=x_T.device)
        for time_step in reversed(range(self.time_steps)):
            if time_step == self.time_steps - 1:
                z[:, :3] = x_T

            # with torch.no_grad():
            # create time steps
            t = torch.ones((x_T.shape[0], ), dtype=torch.long, device=x_T.device) * round(time_step * FLAGS.T / self.time_steps)
            # create noise
            noise = torch.randn_like(x_T, device=x_T.device)

            eps = model(z[:, :3], t)

            z = torch.cat([z[:, :3], eps, noise], dim=1)
            z = self.sampler(time_step, z)

        x_0 = z[:, :3]
        return x_0

class SamplerTimestepLayer(nn.Module):
    def __init__(self, kernel_size, out_channels) -> None:
        super(SamplerTimestepLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=False)
        # self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=9, kernel_size=kernel_size, padding=kernel_size//2)
        # self.bn2 = nn.BatchNorm2d(9)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        # out = F.relu(out, inplace=False)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = F.relu(out, inplace=False)

        return out + residual

class Sampler(nn.Module):
    def __init__(self, time_steps=20, kernel_size=1, out_channels=3):
        super(Sampler, self).__init__()
        self.layers = nn.ModuleList([SamplerTimestepLayer(kernel_size, out_channels) for _ in range(time_steps)])

    def forward(self, time_step, z_t):
        return self.layers[time_step](z_t)
