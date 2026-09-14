import torch
from torch import nn


class Mix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.n = nn.LayerNorm(dim)
        self.a = nn.Linear(dim, dim * 2)
        self.b = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.c = nn.Linear(dim, dim)

    def forward(self, x):
        y = self.n(x)
        a, g = self.a(y).chunk(2, dim=-1)
        a = self.b(a.transpose(1, 2)).transpose(1, 2)
        a = a * torch.sigmoid(g)
        return x + self.c(a)


class Block(nn.Module):
    def __init__(self, dim, head, ff):
        super().__init__()
        self.a = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=head,
            dim_feedforward=ff,
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.b = Mix(dim)

    def forward(self, x, pad=None):
        x = self.a(x, src_key_padding_mask=pad)
        x = self.b(x)
        return x


class Encoder(nn.Module):
    def __init__(self, hdim, ydim=2, dim=128, lat=128, layers=3, head=4, ff=256):
        super().__init__()
        self.h = nn.Linear(hdim, dim)
        self.y = nn.Linear(ydim, dim)
        self.blocks = nn.ModuleList([Block(dim, head, ff) for _ in range(layers)])
        self.n = nn.LayerNorm(dim)
        self.mu = nn.Linear(dim, lat)
        self.logvar = nn.Linear(dim, lat)

        # REPLACE ZERO INIT WITH XAVIER (Phase 1 Task 1A)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.xavier_uniform_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, h, y, sample=False, context_mask=None, future_mask=None):
        if h.ndim != 3 or y.ndim != 3:
            raise ValueError("bad encoder input")
        proj_dtype = self.h.weight.dtype
        h = h.to(dtype=proj_dtype)
        y = y.to(dtype=proj_dtype)
        h = self.h(h)
        y = self.y(y)
        x = torch.cat([h, y], dim=1)
        pad = None
        if context_mask is not None:
            if context_mask.shape != h.shape[:2]:
                raise ValueError("bad context mask")
            if future_mask is None:
                future_mask = torch.ones(y.shape[:2], dtype=torch.bool, device=y.device)
            if future_mask.shape != y.shape[:2]:
                raise ValueError("bad future mask")
            valid = torch.cat([context_mask.to(torch.bool), future_mask.to(torch.bool)], dim=1)
            pad = ~valid
        for block in self.blocks:
            x = block(x, pad=pad)
        x = self.n(x)
        if pad is None:
            x = x.mean(dim=1)
        else:
            valid = (~pad).to(x.dtype).unsqueeze(-1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min_(1.0)
        mu = self.mu(x)
        # Clamp logvar to prevent KL explosion
        logvar = torch.clamp(self.logvar(x), min=-10.0, max=10.0)
        if sample:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        return {"z": z, "mu": mu, "logvar": logvar}
