import torch
from torch import nn

"""
Graph contract:
- feat: (n, 50, d), with xy in feat[..., :2]
- graph["temporal"]: {
    "src_agent": ...,
    "src_time": ...,
    "dst_agent": ...,
    "dst_time": ...,
  }
- graph["spatial"]: same format
- graph["cross"]: same format

Nodes are (agent_id, timestep). Relative geometry is computed from source and
destination nodes, encoded with GRBF plus circular harmonics, and aggregated
back to destination nodes without collapsing the time dimension.
"""


class Mlp(nn.Module):
    def __init__(self, din, dout, hid=None):
        super().__init__()
        if hid is None:
            hid = dout
        self.net = nn.Sequential(
            nn.Linear(din, hid),
            nn.SiLU(),
            nn.Linear(hid, dout),
        )

    def forward(self, x):
        x = x.to(dtype=self.net[0].weight.dtype)
        return self.net(x)


class Mamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.n = nn.LayerNorm(dim)
        self.u = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)
        self.c = nn.Linear(dim, dim)
        self.a = nn.Parameter(torch.zeros(dim))
        self.d = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.to(dtype=self.n.weight.dtype)
        x = self.n(x)
        u = self.u(x)
        g = torch.sigmoid(self.g(x))
        b = torch.nn.functional.softplus(self.b(x))
        c = self.c(x)
        
        # INTERNAL COMPUTATION IN FP32
        b_f32 = b.float()
        u_f32 = u.float()
        c_f32 = c.float()
        a_f32 = torch.nn.functional.softplus(self.a).view(1, 1, -1).float()
        d_f32 = self.d.view(1, 1, -1).float()
        
        step_u_f32 = b_f32 * u_f32
        dec_f32 = torch.exp(-b_f32 * a_f32)
        prefix_f32 = torch.cumprod(dec_f32, dim=1)
        state_f32 = prefix_f32 * torch.cumsum(step_u_f32 / prefix_f32.clamp_min(1e-8), dim=1)
        y_f32 = c_f32 * state_f32 + d_f32 * u_f32
        
        y = y_f32.to(orig_dtype)
        return g * y


class MapEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.type_emb = nn.Embedding(20, 16)
        self.mlp = nn.Sequential(
            nn.Linear(21, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, M, 6] -> [x, y, cos, sin, type, is_int]
        pos = x[..., :2]
        heading = x[..., 2:4]
        ltype = x[..., 4].long()
        is_int = x[..., 5:6]
        t_feat = self.type_emb(ltype)
        feat = torch.cat([pos, heading, t_feat, is_int], dim=-1)
        return self.mlp(feat)


class Geo(nn.Module):
    def __init__(self, dim, rad=16, harm=4, rmax=60.0):
        super().__init__()
        self.rad = int(rad)
        self.harm = int(harm)
        self.rmax = float(rmax)
        cen = torch.linspace(0.0, self.rmax, self.rad)
        wid = self.rmax / max(self.rad - 1, 1)
        self.register_buffer("cen", cen)
        self.register_buffer("sig", torch.tensor(max(wid * 1.5, 1e-3)))
        self.rscale = 1.0
        self.hscale = 0.25
        self.proj = Mlp(self.rad + self.harm * 2, dim)

    def basis(self, rel):
        dx = rel[..., 0]
        dy = rel[..., 1]
        rr = torch.sqrt(dx * dx + dy * dy + 1e-9)
        th = torch.atan2(dy, dx)
        cen = self.cen.view(*([1] * rr.ndim), -1)
        sig = self.sig.view(*([1] * rr.ndim), 1)
        grbf = torch.exp(-0.5 * ((rr.unsqueeze(-1) - cen) / sig) ** 2)
        grbf = grbf / grbf.sum(dim=-1, keepdim=True).clamp_min_(1e-6)
        harmonics = torch.arange(1, self.harm + 1, device=th.device, dtype=th.dtype)
        angles = th.unsqueeze(-1) * harmonics
        har = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).flatten(start_dim=-2)
        return torch.cat([grbf * self.rscale, har * self.hscale], dim=-1)

    def forward(self, rel):
        return self.proj(self.basis(rel))


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = Mlp(dim, dim)
        self.b = Mlp(dim, dim)

    def forward(self, h, g):
        g = g.to(dtype=self.a.net[0].weight.dtype)
        return h * torch.sigmoid(self.a(g)) + self.b(g)


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, din, dim=128, rad=16, harm=4, rmax=60.0, mode="full"):
        super().__init__()
        self.dim = dim
        self.mode = str(mode)
        if self.mode not in {"full", "weak"}:
            raise ValueError(f"unsupported encoder mode: {self.mode}")
        self.inp = Mlp(din, dim)
        self.tmp = Mamba(dim)
        self.map_enc = MapEncoder(dim)
        self.map_ln = nn.LayerNorm(dim)
        self.map_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.map_gate = nn.Linear(dim, dim)
        # PREVENT MAP GATE COLLAPSE (Phase 4 Task 4B)
        nn.init.constant_(self.map_gate.bias, 1.0)
        
        self.tproj = Mlp(dim, dim)
        self.geo = Geo(dim, rad=rad, harm=harm, rmax=rmax)
        self.fuse = Gate(dim)
        self.mix = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32))

    def edge(self, graph, key):
        if graph is None:
            return None
        if not isinstance(graph, dict) or key not in graph:
            raise ValueError("bad graph")
        edge = graph[key]
        need = ["src_node", "dst_node"]
        if not isinstance(edge, dict) or any(x not in edge for x in need):
            raise ValueError("bad graph edge")
        return edge["src_node"].to(torch.long), edge["dst_node"].to(torch.long)

    def agg(self, msg, dst_node, node_count):
        # msg: [E, D], dst_node: [E]
        # node_count: N
        # Final index N is the sink/garbage node
        d = msg.shape[-1]
        out = msg.new_zeros((node_count + 1, d))
        out.index_add_(0, dst_node, msg)
        cnt = msg.new_zeros((node_count + 1, 1))
        one = msg.new_ones((msg.shape[0], 1))
        cnt.index_add_(0, dst_node, one)
        # Division and slicing are vectorized and asynchronous
        return (out / cnt.clamp_min_(1.0))[:node_count]

    def forward(self, feat, graph):
        if feat.ndim != 4:
            raise ValueError("bad features")
        if not isinstance(graph, dict):
            raise ValueError("bad graph")
        hist_mask = graph.get("history_mask")
        if hist_mask is None or hist_mask.shape != feat.shape[:3]:
            raise ValueError("missing history mask")
        bsz, agents, steps = feat.shape[:3]
        node_count = bsz * agents * steps
        node_mask = hist_mask.to(feat.dtype).unsqueeze(-1)
        pos = feat[..., :2]
        x = self.inp(feat)
        x = x * node_mask
        htemp = self.tmp(x.view(bsz * agents, steps, self.dim)).view(bsz, agents, steps, self.dim)
        htemp = htemp * node_mask

        # Map-to-Agent Interaction (T=49)
        map_feat = graph.get("map_feat")
        if map_feat is not None:
            map_mask = graph.get("map_mask")
            # agent last position and heading (t=49)
            agent_last = feat[:, :, -1, :]                # [B, A, D]
            agent_xy = agent_last[..., :2]                # [B, A, 2]
            agent_theta = agent_last[..., 2:3]            # [B, A, 1]

            # map_feat: [B, M, 6] = [x, y, cos, sin, type, is_int]
            map_xy = map_feat[..., :2]                    # [B, M, 2]
            map_heading = map_feat[..., 2:4]              # [B, M, 2]

            # expand for broadcasting
            map_xy = map_xy[:, None, :, :]                # [B, 1, M, 2]
            agent_xy = agent_xy[:, :, None, :]            # [B, A, 1, 2]

            rel_xy = map_xy - agent_xy                    # [B, A, M, 2]

            map_theta = torch.atan2(map_heading[..., 1], map_heading[..., 0])  # [B, M]
            map_theta = map_theta[:, None, :, None]                           # [B, 1, M, 1]
            agent_theta = agent_theta[:, :, None, :]                          # [B, A, 1, 1]

            rel_theta = torch.atan2(
                torch.sin(map_theta - agent_theta),
                torch.cos(map_theta - agent_theta)
            )
            rel_cos = torch.cos(rel_theta)
            rel_sin = torch.sin(rel_theta)

            # keep semantic features unchanged
            lane_type = map_feat[..., 4:5][:, None, :, :].expand(bsz, agents, -1, -1)
            is_int = map_feat[..., 5:6][:, None, :, :].expand(bsz, agents, -1, -1)

            map_feat_rel = torch.cat([rel_xy, rel_cos, rel_sin, lane_type, is_int], dim=-1)
            # shape: [B, A, M, 6]
            B, A, M, _ = map_feat_rel.shape
            map_feat_rel = map_feat_rel.view(B * A, M, 6)

            if map_mask is not None:
                map_mask = map_mask[:, None, :].expand(B, A, M).reshape(B * A, M)

            # 1. Encode map tokens
            map_tokens = self.map_enc(map_feat_rel)
            # 2. Query preparation
            h_last = htemp[:, :, -1, :]                 # [B, A, D]
            h_query = self.map_ln(h_last)
            h_query = h_query.reshape(B * A, 1, self.dim)

            if map_mask is not None:
                # Add one valid entry for the sink node to prevent all-masked nan
                map_mask_with_sink = torch.cat([map_mask, map_mask.new_ones((B * A, 1))], dim=1)
                map_tokens_with_sink = torch.cat([map_tokens, map_tokens.new_zeros((B * A, 1, self.dim))], dim=1)
                
                h_mapaware, _ = self.map_attn(
                    query=h_query,
                    key=map_tokens_with_sink,
                    value=map_tokens_with_sink,
                    key_padding_mask=~map_mask_with_sink
                )
            else:
                h_mapaware, _ = self.map_attn(
                    query=h_query,
                    key=map_tokens,
                    value=map_tokens
                )
            # 4. Gated fusion ONLY on final observed timestep
            h_mapaware = h_mapaware.view(B, A, self.dim)
            g = torch.sigmoid(self.map_gate(h_last))
            gated_map = g * h_mapaware
            
            htemp_last = htemp[:, :, -1, :] + gated_map
            htemp = torch.cat([htemp[:, :, :-1, :], htemp_last.unsqueeze(2)], dim=2)
            htemp = htemp * node_mask

        if self.mode == "weak":
            return htemp
        
        # Add Sink Node row to prevent CUDA out-of-bounds on masked edges
        hflat = torch.cat([htemp.view(node_count, self.dim), htemp.new_zeros((1, self.dim))], dim=0)
        pflat = torch.cat([pos.view(node_count, 2), pos.new_zeros((1, 2))], dim=0)

        tedge = self.edge(graph, "temporal")
        if tedge is None:
            hflow = htemp.new_zeros(htemp.shape)
        else:
            src_node, dst_node = tedge
            tmsg = self.tproj(hflat[src_node])
            hflow = self.agg(tmsg, dst_node, node_count).view(bsz, agents, steps, self.dim)
        sedge = self.edge(graph, "spatial")
        if sedge is None:
            hsp = htemp.new_zeros(htemp.shape)
        else:
            src_node, dst_node = sedge
            sgeo = self.geo(pflat[src_node] - pflat[dst_node])
            hsp = self.agg(sgeo, dst_node, node_count).view(bsz, agents, steps, self.dim)
        cedge = self.edge(graph, "cross")
        if cedge is None:
            hst = htemp.new_zeros(htemp.shape)
        else:
            src_node, dst_node = cedge
            cgeo = self.geo(pflat[src_node] - pflat[dst_node])
            ctmp = hflat[src_node]
            cmsg = self.fuse(ctmp, cgeo)
            hst = self.agg(cmsg, dst_node, node_count).view(bsz, agents, steps, self.dim)
        w = torch.softmax(self.mix, dim=0).to(htemp.dtype)
        out = w[0] * htemp + w[1] * hflow + w[2] * hsp + w[3] * hst
        return out * node_mask
