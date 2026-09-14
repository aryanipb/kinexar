import torch


def repeat_batch_rows(x, repeat_count):
    return x.repeat_interleave(repeat_count, dim=0)


def expand_to_match(x, ref):
    while x.ndim < ref.ndim:
        x = x.unsqueeze(1)
    return x


def to_abs_coordinates(y_local, anchor, scale):
    anchor = expand_to_match(anchor, y_local).to(dtype=y_local.dtype)
    scale = expand_to_match(scale, y_local).to(dtype=y_local.dtype)
    return y_local * scale + anchor


def to_local_coordinates(y_abs, anchor, scale):
    anchor = expand_to_match(anchor, y_abs).to(dtype=y_abs.dtype)
    scale = expand_to_match(scale, y_abs).to(dtype=y_abs.dtype)
    return (y_abs - anchor) / scale


def build_latent_generator(seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def sample_latent_tensor(lat_dim, batch, count, device, dtype, mu=None, logvar=None, generator=None):
    if mu is not None and logvar is not None:
        # Use explicit dimensions instead of tuple to help torch.compile/dynamo
        # mu shape is (batch, lat_dim)
        # result shape is (batch, count, lat_dim)
        eps = torch.empty((batch, count, lat_dim), device=mu.device, dtype=mu.dtype).normal_(generator=generator)
        return mu.unsqueeze(1) + torch.exp(0.5 * logvar).unsqueeze(1) * eps
    return torch.empty((batch, count, lat_dim), device=device, dtype=dtype).normal_(generator=generator)
