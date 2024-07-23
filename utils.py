import torch
from torchdiffeq import odeint
from comfy.utils import ProgressBar
from tqdm import tqdm

def get_2d_rotary_pos_embed_lumina(head_dim, height, width, linear_factor=1.0, ntk_factor=1.0):
    x = torch.arange(width).float() * linear_factor
    y = torch.arange(height).float() * linear_factor
    x, y = torch.meshgrid(x, y, indexing="ij")
    pos = torch.stack([x, y], dim=-1)
    pos = pos.view(-1, 2)

    dim = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid_inp = torch.einsum("i,j->ij", pos.view(-1), inv_freq)
    sinusoid_inp = sinusoid_inp * ntk_factor

    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    sin_pos = torch.cat([sin, sin], dim=-1).view(height, width, head_dim)
    cos_pos = torch.cat([cos, cos], dim=-1).view(height, width, head_dim)

    return sin_pos, cos_pos

class ODE:
    def __init__(self, num_steps, sampler_type="midpoint", time_shifting_factor=None, strength=1.0, t0=0.0, t1=1.0):
        self.t = torch.linspace(t0, t1, num_steps)
        if time_shifting_factor:
            self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)

        if strength != 1.0:
            self.t = self.t[int(num_steps * (1 - strength)):]

        self.sampler_type = sampler_type
        if self.sampler_type == "euler":
            total_steps = len(self.t)
        else:
            total_steps = (len(self.t) * 2) - 2
        self.comfy_pbar = ProgressBar(total_steps)
        self.pbar = tqdm(total=total_steps, desc='ODE Sampling')

    def sample(self, model_fn, **model_kwargs):
        device = next(model_fn.parameters()).device

        def _fn(t, x):
            t = torch.ones(x.size(0)).to(device) * t
            model_output = model_fn(x, t, **model_kwargs)
            self.pbar.update(1)
            self.comfy_pbar.update(1)
            return model_output

        t = self.t.to(device)
        x = model_kwargs.pop('x')
        samples = odeint(_fn, x, t, method=self.sampler_type)
        self.pbar.close()
        return samples
