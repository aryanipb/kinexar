import torch


def vals(inp, kind):
    if inp.shape[-1] < 4:
        raise ValueError("bad shape")
    x = inp[..., 0]
    y = inp[..., 1]
    if kind == "xyvh":
        v = inp[..., 2]
        t = inp[..., 3]
        vx = v * torch.cos(t)
        vy = v * torch.sin(t)
        return x, y, vx, vy, t
    if kind == "xyhvv":
        t = inp[..., 2]
        vx = inp[..., 3]
        vy = inp[..., 4]
        return x, y, vx, vy, t
    if kind == "xyvvh":
        vx = inp[..., 2]
        vy = inp[..., 3]
        t = inp[..., 4]
        return x, y, vx, vy, t
    raise ValueError(kind)


def pick(inp):
    if inp.shape[-1] == 4:
        return "xyvh"
    a = inp[..., 2].abs().amax().item()
    b = inp[..., 4].abs().amax().item()
    if inp.shape[-1] >= 5 and a <= 4.0 and b > 4.0:
        return "xyhvv"
    return "xyvvh"


def ang(vx, vy, msk, eps=1e-3):
    shp = vx.shape
    out = torch.zeros(shp, dtype=vx.dtype, device=vx.device)
    spd = torch.sqrt(vx * vx + vy * vy)
    for i in range(shp[0]):
        last = torch.zeros((), dtype=vx.dtype, device=vx.device)
        for t in range(shp[1]):
            if not bool(msk[i, t]):
                continue
            if float(spd[i, t]) >= eps:
                last = torch.atan2(vy[i, t], vx[i, t])
            out[i, t] = last
    return out


def run(inp, msk, dt=0.1, kind="auto"):
    if kind == "auto":
        kind = pick(inp)
    x, y, vx, vy, _ = vals(inp, kind)
    t = ang(vx, vy, msk)
    v = torch.sqrt(vx * vx + vy * vy)
    a = torch.zeros_like(v)
    a[:, 1:] = (v[:, 1:] - v[:, :-1]) / dt
    ax = a * torch.cos(t)
    ay = a * torch.sin(t)
    out = torch.stack([x, y, vx, vy, ax, ay, t], dim=-1)
    z = msk.unsqueeze(-1).to(out.dtype)
    return out * z
