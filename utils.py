import torch
import math

def get_2d_rotary_pos_embed_lumina(head_dim, height, width, linear_factor=1.0, ntk_factor=1.0):
    x = torch.arange(width).float() * linear_factor
    y = torch.arange(height).float() * linear_factor
    x, y = torch.meshgrid(x, y, indexing="ij")
    pos = torch.stack([x, y], dim=-1)
    pos = pos.view(-1, 2)

    # Adjust the dimension to match head_dim
    dim = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid_inp = torch.einsum("i,j->ij", pos.view(-1), inv_freq)
    sinusoid_inp = sinusoid_inp * ntk_factor

    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    sin_pos = torch.cat([sin, sin], dim=-1).view(height, width, head_dim)
    cos_pos = torch.cat([cos, cos], dim=-1).view(height, width, head_dim)

    return sin_pos, cos_pos

def apply_time_aware_scaling(scheduler, timestep, scaling_watershed, scaling_factor):
    if torch.is_tensor(timestep):
        current_timestep = 1 - timestep / scheduler.config.num_train_timesteps
        linear_factor = torch.where(current_timestep < scaling_watershed, scaling_factor, 1.0)
        ntk_factor = torch.where(current_timestep < scaling_watershed, 1.0, scaling_factor)
    else:
        current_timestep = 1 - timestep / scheduler.config.num_train_timesteps
        if current_timestep < scaling_watershed:
            linear_factor = scaling_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scaling_factor
        linear_factor = torch.tensor([linear_factor])
        ntk_factor = torch.tensor([ntk_factor])
    return linear_factor, ntk_factor
