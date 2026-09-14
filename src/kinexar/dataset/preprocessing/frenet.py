import math

import torch


def mid(left, right):
    n = min(left.shape[0], right.shape[0])
    if n < 2:
        return None, None
    left = left[:n]
    right = right[:n]
    cen = (left + right) * 0.5
    wid = torch.sqrt(((left - right) ** 2).sum(dim=-1))
    return cen, wid


def segs(cen, wid):
    a = cen[:-1]
    b = cen[1:]
    v = b - a
    l = torch.sqrt((v * v).sum(dim=-1)).clamp_min(1e-6)
    s = torch.zeros(cen.shape[0], dtype=cen.dtype, device=cen.device)
    s[1:] = torch.cumsum(l, dim=0)
    w = (wid[:-1] + wid[1:]) * 0.5
    return a, v, l, s[:-1], w


def diff(a, b):
    d = abs(a - b)
    p = 2 * math.pi
    return min(d, p - d)


def first(pos, msk):
    ids = torch.nonzero(msk, as_tuple=False).flatten()
    if ids.numel() == 0:
        return None
    t0 = int(ids[0])
    p0 = pos[t0]
    h0 = 0.0
    for t1 in ids[1:].tolist():
        dv = pos[t1] - p0
        n = float(torch.sqrt((dv * dv).sum()))
        if n >= 1e-3:
            h0 = math.atan2(float(dv[1]), float(dv[0]))
            break
    return t0, p0, h0


def proj(pos, a, v, l, s0, w0):
    q = pos[:, None, :] - a[None, :, :]
    den = (v * v).sum(dim=-1).clamp_min(1e-6)
    t = (q * v[None, :, :]).sum(dim=-1) / den[None, :]
    t = t.clamp(0.0, 1.0)
    p = a[None, :, :] + t[..., None] * v[None, :, :]
    d = pos[:, None, :] - p
    n = (d * d).sum(dim=-1)
    k = torch.argmin(n, dim=-1)
    i = torch.arange(pos.shape[0], device=pos.device)
    tk = t[i, k]
    dk = d[i, k]
    vk = v[k]
    lk = l[k]
    sk = s0[k] + tk * lk
    wk = w0[k]
    z = vk[:, 0] * dk[:, 1] - vk[:, 1] * dk[:, 0]
    dd = torch.sign(z) * torch.sqrt((dk * dk).sum(dim=-1))
    ang = torch.atan2(vk[:, 1], vk[:, 0])
    near = torch.sqrt(n[i, k])
    return sk, dd, wk, ang, near


def one(pos, msk, left, right):
    cen, wid = mid(left, right)
    out = torch.zeros((pos.shape[0], 2), dtype=pos.dtype, device=pos.device)
    lan = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)
    val = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
    if cen is None:
        return out, lan, val
    a, v, l, s0, w0 = segs(cen, wid)
    ids = torch.nonzero(msk, as_tuple=False).flatten()
    if ids.numel() == 0:
        return out, lan, val
    pts = pos[ids]
    s, d, w, _, _ = proj(pts, a, v, l, s0, w0)
    out[ids, 0] = s
    out[ids, 1] = d
    lan[ids] = w
    val[ids] = True
    return out, lan, val


def score(pos, msk, lane, rad, turn):
    firsts = first(pos, msk)
    if firsts is None:
        return None
    _, p0, h0 = firsts
    left, right = lane
    cen, wid = mid(left, right)
    if cen is None:
        return None
    a, v, l, s0, w0 = segs(cen, wid)
    _, _, _, ang, near = proj(p0[None, :], a, v, l, s0, w0)
    near = float(near[0])
    if near > rad:
        return None
    err = diff(h0, float(ang[0]))
    scr = near + turn * err
    return scr, near, err


def choose(pos, msk, lanes, rad=8.0, turn=2.0, far=1.5):
    cand = []
    for i, lane in enumerate(lanes):
        cur = score(pos, msk, lane, rad, turn)
        if cur is None:
            continue
        cand.append((cur[0], cur[1], cur[2], i))
    if not cand:
        return None
    cand.sort()
    pick = None
    best = None
    for _, _, _, i in cand:
        out, lan, val = one(pos, msk, lanes[i][0], lanes[i][1])
        bad = torch.zeros_like(val)
        ids = val & (lan > 0)
        bad[ids] = out[ids, 1].abs() > (lan[ids] * far)
        cnt = int(bad.sum())
        cur = (cnt, i, out, lan, val & ~bad)
        if best is None or cnt < best[0]:
            best = cur
        if cnt == 0:
            pick = cur
            break
    if pick is None:
        pick = best
    return pick[2], pick[3], pick[4], pick[1]


def run(pos, left, right, msk=None):
    if msk is None:
        msk = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
    out = torch.zeros((*pos.shape[:2], 2), dtype=pos.dtype, device=pos.device)
    lan = torch.zeros(pos.shape[:2], dtype=pos.dtype, device=pos.device)
    val = torch.zeros(pos.shape[:2], dtype=torch.bool, device=pos.device)
    tmp, w, good = one(pos[0], msk[0], left, right)
    out[0] = tmp
    lan[0] = w
    val[0] = good
    return {"frenet": out, "lane_width": lan, "valid": val}


def runset(pos, lanes, msk=None, rad=8.0, turn=2.0, far=1.5):
    if msk is None:
        msk = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
    out = torch.zeros((*pos.shape[:2], 2), dtype=pos.dtype, device=pos.device)
    lan = torch.zeros(pos.shape[:2], dtype=pos.dtype, device=pos.device)
    val = torch.zeros(pos.shape[:2], dtype=torch.bool, device=pos.device)
    ids = []
    for i in range(pos.shape[0]):
        cur = choose(pos[i], msk[i], lanes, rad=rad, turn=turn, far=far)
        if cur is None:
            ids.append(-1)
            continue
        tmp, w, good, lid = cur
        out[i] = tmp
        lan[i] = w
        val[i] = good
        ids.append(lid)
    return {"frenet": out, "lane_width": lan, "valid": val, "lane_id": ids}
