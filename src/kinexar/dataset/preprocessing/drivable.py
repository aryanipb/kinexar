import math

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


def arr(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float16)


class Road:
    def __init__(self, polys, res=0.3, pad=5.0):
        self.res = float(res)
        self.pad = float(pad)
        self.polys = [arr(x)[:, :2] for x in polys if len(x) >= 3]
        if not self.polys:
            self.ok = False
            return
        pts = np.concatenate(self.polys, axis=0)
        self.x0 = float(pts[:, 0].min() - self.pad)
        self.y0 = float(pts[:, 1].min() - self.pad)
        self.x1 = float(pts[:, 0].max() + self.pad)
        self.y1 = float(pts[:, 1].max() + self.pad)
        self.w = int(math.ceil((self.x1 - self.x0) / self.res)) + 1
        self.h = int(math.ceil((self.y1 - self.y0) / self.res)) + 1
        img = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(img)
        for poly in self.polys:
            xy = []
            for x, y in poly:
                px = (float(x) - self.x0) / self.res
                py = (self.y1 - float(y)) / self.res
                xy.append((px, py))
            draw.polygon(xy, fill=1)
        mask = np.array(img, dtype=np.uint8) > 0
        dist = distance_transform_edt(~mask, sampling=self.res).astype(np.float16)
        self.mask = torch.from_numpy(mask)
        self.dist = torch.from_numpy(dist)
        self.ok = True

    def check(self, pts):
        if not self.ok:
            shp = pts.shape[:-1]
            return torch.full(shp, float("inf"), dtype=pts.dtype, device=pts.device)
        x = pts[..., 0]
        y = pts[..., 1]
        px = (x - self.x0) / self.res
        py = (self.y1 - y) / self.res
        ix = torch.floor(px).to(torch.long)
        iy = torch.floor(py).to(torch.long)
        ox0 = torch.clamp(-px, min=0.0)
        oy0 = torch.clamp(-py, min=0.0)
        ox1 = torch.clamp(px - (self.w - 1), min=0.0)
        oy1 = torch.clamp(py - (self.h - 1), min=0.0)
        ext = torch.sqrt((ox0 + ox1) ** 2 + (oy0 + oy1) ** 2) * self.res
        ix = ix.clamp(0, self.w - 1)
        iy = iy.clamp(0, self.h - 1)
        dev = pts.device
        mask = self.mask.to(dev)
        dist = self.dist.to(dev)
        hit = mask[iy, ix]
        out = dist[iy, ix] + ext.to(dist.dtype)
        out = torch.where(hit, torch.zeros_like(out), out)
        return out.to(pts.dtype)
