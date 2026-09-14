import torch
from torch import nn


class Prior(nn.Module):
    def __init__(self, hdim, lat=128, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hdim * 2, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hid, lat)
        self.logvar = nn.Linear(hid, lat)
        
        # Zero-Initialize for safe start
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, h_global, ctx_mean):
        if h_global.ndim != 2 or ctx_mean.ndim != 2 or h_global.shape != ctx_mean.shape:
            raise ValueError("bad prior input")
        x = torch.cat([h_global, ctx_mean], dim=-1)
        x = x.to(dtype=self.net[0].weight.dtype)
        h = self.net(x)
        mu = self.mu(h)
        # Clamp logvar to prevent KL explosion
        logvar = torch.clamp(self.logvar(h), min=-10.0, max=10.0)
        return {"mu": mu, "logvar": logvar}
