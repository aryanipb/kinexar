import torch
import torch.nn.functional as F
from torch import nn

from kinexar.models.cvae.encoder import Block


DEFAULT_STRUCTURED_EXPERTS = (
    "cv",
    "ca",
    "ctra",
    "jerk",
    "jerk_beta",
    "jerk_turn",
)

STRUCTURED_PARAM_DIMS = {
    "cv": 2,
    "ca": 2,
    "ctra": 2,
    "jerk": 1,
    "jerk_beta": 2,
    "jerk_turn": 2,
}


class Mlp(nn.Module):
    def __init__(self, din, dout, hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, dout),
        )

    def forward(self, x):
        x = x.to(dtype=self.net[0].weight.dtype)
        return self.net(x)


class ResidualRefinement(nn.Module):
    def __init__(self, c_dim, d_dim, z_dim, h_dim=128):
        super().__init__()
        self.d1 = nn.Sequential(
            nn.Linear(c_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
        )
        self.d2 = nn.Sequential(
            nn.Linear(d_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
        )
        self.d3 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
        )
        self.w_g = nn.Linear(h_dim, h_dim)
        self.w_r = nn.Linear(h_dim, d_dim)

    def forward(self, theta, y_coarse, z, lambd):
        # theta: [B, K, T, C]
        # y_coarse: [B, K, T, D]
        # z: [B, K, Z]
        h1 = self.d1(theta)
        h2 = self.d2(y_coarse)
        h3 = self.d3(z)                             # [B, K, H]
        h3 = h3.unsqueeze(2)                        # [B, K, 1, H]
        g = torch.sigmoid(self.w_g(h1))
        fused = g * h2 + h3
        rk = self.w_r(fused)
        return y_coarse + lambd * rk


class MoEPhysicsDecoder(nn.Module):
    def __init__(
        self,
        hdim,
        lat=128,
        dim=128,
        steps=60,
        expert_count=6,
        expert_names=None,
        use_transformer_coeff_decoder=True,
        use_mlp_coeff_decoder=False,
        use_structured_latent=True,
        latent_group_dim=16,
        decoder_layers=3,
        decoder_heads=4,
        use_residual_refinement=False,
        refinement_lambda=0.35,
    ):
        super().__init__()
        self.steps = int(steps)
        self.expert_names = list(expert_names or DEFAULT_STRUCTURED_EXPERTS)
        self.experts = len(self.expert_names)
        if self.experts < 1:
            raise ValueError("MoEPhysicsDecoder requires at least one expert")
        self.dt = 0.1
        self.lat = int(lat)
        self.dim = int(dim)
        self.use_transformer_coeff_decoder = bool(use_transformer_coeff_decoder)
        self.use_mlp_coeff_decoder = bool(use_mlp_coeff_decoder)
        self.use_structured_latent = bool(use_structured_latent and lat % 4 == 0)
        self.latent_group_dim = int(latent_group_dim)
        self.decoder_layers = max(int(decoder_layers), 1)
        self.decoder_heads = int(decoder_heads)
        self.use_residual_refinement = bool(use_residual_refinement)
        self.refinement_lambda = float(refinement_lambda)
        if self.use_transformer_coeff_decoder and self.use_mlp_coeff_decoder:
            raise ValueError("choose one coefficient decoder: transformer or mlp")
        if not self.use_transformer_coeff_decoder and not self.use_mlp_coeff_decoder:
            self.use_transformer_coeff_decoder = True
        hid = dim * 2
        self.scene_proj = nn.Linear(hdim, dim)
        self.expert_embed = nn.Embedding(self.experts, dim)
        self.latent_add_proj = nn.Linear(lat, dim)
        self.latent_gate_proj = nn.Linear(lat, dim)
        if self.use_structured_latent:
            group_width = lat // 4
            self.latent_group_proj = nn.ModuleList([nn.Linear(group_width, dim) for _ in range(4)])
        else:
            self.latent_group_proj = None
        self.coeff_dims = {name: self._expanded_coeff_dim(STRUCTURED_PARAM_DIMS[name]) for name in self.expert_names}
        if self.use_transformer_coeff_decoder:
            self.token_blocks = nn.ModuleList([Block(dim, self.decoder_heads, hid * 2) for _ in range(self.decoder_layers)])
            self.token_norm = nn.LayerNorm(dim)
            self.coeff_basis_heads = nn.ModuleList([nn.Linear(dim, self.coeff_dims[name]) for name in self.expert_names])
        else:
            inp = hdim + lat
            self.param_heads = nn.ModuleList([Mlp(inp, self.coeff_dims[name], hid) for name in self.expert_names])
        self.raw_heads = nn.ModuleList([nn.Linear(self.coeff_dims[name], self.steps * STRUCTURED_PARAM_DIMS[name]) for name in self.expert_names])

        # REPLACE ZERO INIT WITH XAVIER (Phase 1 Task 1A)
        for head in self.raw_heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

        if self.use_residual_refinement:
            # Max C is 2 for current experts
            self.refiner = ResidualRefinement(c_dim=2, d_dim=2, z_dim=lat, h_dim=128)
            nn.init.xavier_uniform_(self.refiner.w_r.weight)
            if self.refiner.w_r.bias is not None:
                nn.init.zeros_(self.refiner.w_r.bias)
        else:
            self.refiner = None

    def _expanded_coeff_dim(self, raw_dim):
        return max(int(raw_dim) * 4, 6)

    def _structured_latent_features(self, z):
        if not self.use_structured_latent:
            return z.new_zeros((z.shape[0], z.shape[1], self.dim))
        groups = torch.chunk(z, 4, dim=-1)
        parts = [proj(group) for proj, group in zip(self.latent_group_proj, groups)]
        return torch.stack(parts, dim=0).sum(dim=0)

    def _latent_mode(self):
        return "structured" if self.use_structured_latent else "flat"

    def _coeff_tokens(self, h_global, z):
        proj_dtype = self.scene_proj.weight.dtype
        h_global = h_global.to(dtype=proj_dtype)
        z = z.to(dtype=proj_dtype)
        scene_token = self.scene_proj(h_global).unsqueeze(1)
        latent_add = self.latent_add_proj(z) + self._structured_latent_features(z)
        latent_gate = torch.tanh(self.latent_gate_proj(z))
        expert_ids = torch.arange(self.experts, device=z.device, dtype=torch.long)
        expert_token = self.expert_embed(expert_ids).unsqueeze(0)
        # The latent both injects new information and gates the shared scene token.
        tokens = scene_token * (1.0 + 0.5 * latent_gate) + latent_add + expert_token
        for block in self.token_blocks:
            tokens = block(tokens)
        return self.token_norm(tokens)

    def _predict_coefficients(self, h_global, z):
        if self.use_transformer_coeff_decoder:
            tokens = self._coeff_tokens(h_global, z)
            coeff_basis = [head(tokens[:, expert_index]) for expert_index, head in enumerate(self.coeff_basis_heads)]
        else:
            proj_dtype = self.param_heads[0].net[0].weight.dtype
            h_global_exp = h_global.to(dtype=proj_dtype).unsqueeze(1).expand(-1, self.experts, -1)
            hz = torch.cat([h_global_exp, z.to(dtype=proj_dtype)], dim=-1)
            coeff_basis = [head(hz[:, expert_index]) for expert_index, head in enumerate(self.param_heads)]
        
        # Scale the raw Neural Nudges by 1/50 so they match the scaled manifold.
        raw_params = [(head(F.silu(basis)) / 50.0).view(-1, self.steps, STRUCTURED_PARAM_DIMS[name]) for name, (basis, head) in zip(self.expert_names, zip(coeff_basis, self.raw_heads))]
        return coeff_basis, raw_params

    def _coeff_scales(self, coeff_by_name):
        def mean_abs(name):
            if name not in coeff_by_name:
                return None
            return coeff_by_name[name].abs().mean()

        v_scale = []
        if "cv" in coeff_by_name:
            v_scale.append((4.0 * torch.tanh(coeff_by_name["cv"])).abs().mean())
        if "ca" in coeff_by_name:
            a_scale = (2.0 * torch.tanh(coeff_by_name["ca"])).abs().mean()
        else:
            a_scale = None
        omega_vals = []
        if "ctra" in coeff_by_name:
            omega_vals.append((0.6 * torch.tanh(coeff_by_name["ctra"][..., 1:2])).abs().mean())
        if "jerk_turn" in coeff_by_name:
            omega_vals.append((0.6 * torch.tanh(coeff_by_name["jerk_turn"][..., 1:2])).abs().mean())
        j_vals = []
        if "jerk" in coeff_by_name:
            j_vals.append((1.5 * torch.tanh(coeff_by_name["jerk"][..., :1])).abs().mean())
        if "jerk_beta" in coeff_by_name:
            j_vals.append((1.5 * torch.tanh(coeff_by_name["jerk_beta"][..., :1])).abs().mean())
        if "jerk_turn" in coeff_by_name:
            j_vals.append((1.5 * torch.tanh(coeff_by_name["jerk_turn"][..., :1])).abs().mean())
        alpha_vals = []
        if "jerk_beta" in coeff_by_name:
            alpha_vals.append((1.0 * torch.tanh(coeff_by_name["jerk_beta"][..., 1:2])).abs().mean())
        return {
            "v_scale": sum(v_scale) / len(v_scale) if v_scale else torch.tensor(0.0, device=next(iter(coeff_by_name.values())).device),
            "a_scale": a_scale if a_scale is not None else torch.tensor(0.0, device=next(iter(coeff_by_name.values())).device),
            "omega_scale": sum(omega_vals) / len(omega_vals) if omega_vals else torch.tensor(0.0, device=next(iter(coeff_by_name.values())).device),
            "j_scale": sum(j_vals) / len(j_vals) if j_vals else torch.tensor(0.0, device=next(iter(coeff_by_name.values())).device),
            "alpha_scale": sum(alpha_vals) / len(alpha_vals) if alpha_vals else torch.tensor(0.0, device=next(iter(coeff_by_name.values())).device),
        }

    def _last_indices(self, mask):
        steps = mask.shape[1]
        idx = torch.arange(steps, device=mask.device, dtype=torch.long).view(1, steps)
        last = torch.where(mask, idx, -1).amax(dim=1)
        prev = torch.where(mask & (idx < last.unsqueeze(1)), idx, -1).amax(dim=1)
        prev = torch.where(prev >= 0, prev, last)
        prev2 = torch.where(mask & (idx < prev.unsqueeze(1)), idx, -1).amax(dim=1)
        prev2 = torch.where(prev2 >= 0, prev2, prev)
        return last, prev, prev2

    def _theta_from_vel(self, vel, theta_fallback):
        theta_vel = torch.atan2(vel[..., 1:2], vel[..., 0:1])
        speed = torch.linalg.vector_norm(vel, dim=-1, keepdim=True)
        return torch.where(speed > 1e-4, theta_vel, theta_fallback)

    def root_state(self, feat, anchor, scale, history_mask=None):
        if history_mask is None:
            last_idx = torch.full((feat.shape[0],), feat.shape[2] - 1, device=feat.device, dtype=torch.long)
            prev_idx = torch.clamp(last_idx - 1, min=0)
            prev2_idx = torch.clamp(prev_idx - 1, min=0)
        else:
            if history_mask.ndim != 3 or history_mask.shape != feat.shape[:3]:
                raise ValueError("bad history mask")
            root_mask = history_mask[:, 0]
            last_idx, prev_idx, prev2_idx = self._last_indices(root_mask)
        batch_idx = torch.arange(feat.shape[0], device=feat.device)
        root = feat[:, 0]
        # Data is already relative and scaled in refinery.
        pos = root[batch_idx, last_idx, :2].unsqueeze(1)
        vel = root[batch_idx, last_idx, 3:5].unsqueeze(1)
        prev_vel = root[batch_idx, prev_idx, 3:5].unsqueeze(1)
        prev2_vel = root[batch_idx, prev2_idx, 3:5].unsqueeze(1)
        theta_last = root[batch_idx, last_idx, 2:3].unsqueeze(1)
        theta_prev_feat = root[batch_idx, prev_idx, 2:3].unsqueeze(1)
        theta_prev2_feat = root[batch_idx, prev2_idx, 2:3].unsqueeze(1)
        theta0 = self._theta_from_vel(vel, theta_last)
        theta_prev = self._theta_from_vel(prev_vel, theta_prev_feat)
        theta_prev2 = self._theta_from_vel(prev2_vel, theta_prev2_feat)
        speed = torch.linalg.vector_norm(vel, dim=-1, keepdim=True).clamp_min(0.001)
        prev_speed = torch.linalg.vector_norm(prev_vel, dim=-1, keepdim=True)
        prev2_speed = torch.linalg.vector_norm(prev2_vel, dim=-1, keepdim=True)
        a0 = (speed - prev_speed) / self.dt
        prev_a = (prev_speed - prev2_speed) / self.dt
        j0 = (a0 - prev_a) / self.dt
        omega0 = self._wrap(theta0 - theta_prev) / self.dt
        prev_omega = self._wrap(theta_prev - theta_prev2) / self.dt
        beta0 = (omega0 - prev_omega) / self.dt
        heading = torch.cat([torch.cos(theta0), torch.sin(theta0)], dim=-1)
        return {
            "pos": pos,
            "vel": vel,
            "speed": speed,
            "heading": heading,
            "theta0": theta0,
            "a0": a0,
            "omega0": omega0,
            "j0": j0,
            "beta0": beta0,
        }

    def _wrap(self, angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def time(self, ref):
        t = torch.arange(1, self.steps + 1, device=ref.device, dtype=ref.dtype)
        tt = t.view(1, self.steps, 1) * self.dt
        tt2 = tt * tt
        tt3 = tt2 * tt
        return tt, tt2, tt3

    def constant_velocity(self, state, raw):
        tt, _, _ = self.time(state["pos"])
        vel_seq = state["vel"] + 4.0 * torch.tanh(raw)
        pos = state["pos"] + vel_seq * tt
        feat = torch.cat([vel_seq, raw.new_zeros(vel_seq.shape)], dim=-1)
        return pos, feat, {"vel": vel_seq}

    def constant_acceleration(self, state, raw):
        tt, tt2, _ = self.time(state["pos"])
        acc_seq = 2.0 * torch.tanh(raw)
        vel = state["vel"] + acc_seq * tt
        pos = state["pos"] + state["vel"] * tt + 0.5 * acc_seq * tt2
        feat = torch.cat([acc_seq, raw.new_zeros(acc_seq.shape)], dim=-1)
        return pos, feat, {"vel": vel, "acc": acc_seq}

    def accel_turn(self, state, raw):
        tt, tt2, _ = self.time(state["pos"])
        acc_seq = 2.0 * torch.tanh(raw[..., 0:1])
        omega_seq = 0.6 * torch.tanh(raw[..., 1:2])
        theta_eff = state["theta0"] + 0.5 * omega_seq * tt
        s_t = state["speed"] * tt + 0.5 * acc_seq * tt2
        pos = state["pos"] + s_t * torch.cat([torch.cos(theta_eff), torch.sin(theta_eff)], dim=-1)
        theta_seq = state["theta0"] + omega_seq * tt
        v_next = (state["speed"] + acc_seq * tt).clamp_min(0.001)
        vel = torch.cat([v_next * torch.cos(theta_seq), v_next * torch.sin(theta_seq)], dim=-1)
        feat = torch.cat([acc_seq, omega_seq, state["speed"].expand(-1, self.steps, -1), state["theta0"].expand(-1, self.steps, -1)], dim=-1)
        return pos, feat, {
            "vel": vel,
            "acc": acc_seq,
            "omega": omega_seq,
            "theta": theta_seq,
            "theta0": state["theta0"],
        }

    def changing_acceleration(self, state, raw):
        tt, tt2, tt3 = self.time(state["pos"])
        jerk_seq = 1.5 * torch.tanh(raw[..., 0:1])
        s_t = state["speed"] * tt + 0.5 * state["a0"] * tt2 + (1.0 / 6.0) * jerk_seq * tt3
        pos = state["pos"] + s_t * torch.cat([torch.cos(state["theta0"]), torch.sin(state["theta0"])], dim=-1)
        v_next = (state["speed"] + state["a0"] * tt + 0.5 * jerk_seq * tt2).clamp_min(0.001)
        vel = torch.cat([v_next * torch.cos(state["theta0"]), v_next * torch.sin(state["theta0"])], dim=-1)
        acc_seq = state["a0"] + jerk_seq * tt
        feat = torch.cat([jerk_seq, state["a0"].expand(-1, self.steps, -1), state["speed"].expand(-1, self.steps, -1), state["theta0"].expand(-1, self.steps, -1)], dim=-1)
        return pos, feat, {
            "vel": vel,
            "acc": acc_seq,
            "j": jerk_seq,
        }

    def changing_accel_turn(self, state, raw):
        tt, tt2, tt3 = self.time(state["pos"])
        jerk_seq = 1.5 * torch.tanh(raw[..., 0:1])
        beta_seq = 1.0 * torch.tanh(raw[..., 1:2])
        s_t = state["speed"] * tt + 0.5 * state["a0"] * tt2 + (1.0 / 6.0) * jerk_seq * tt3
        theta_eff = state["theta0"] + 0.5 * state["omega0"] * tt + (1.0 / 6.0) * beta_seq * tt2
        pos = state["pos"] + s_t * torch.cat([torch.cos(theta_eff), torch.sin(theta_eff)], dim=-1)
        v_next = (state["speed"] + state["a0"] * tt + 0.5 * jerk_seq * tt2).clamp_min(0.001)
        theta_seq = state["theta0"] + state["omega0"] * tt + 0.5 * beta_seq * tt2
        vel = torch.cat([v_next * torch.cos(theta_seq), v_next * torch.sin(theta_seq)], dim=-1)
        acc_seq = state["a0"] + jerk_seq * tt
        omega_seq = state["omega0"] + beta_seq * tt
        feat = torch.cat([jerk_seq, beta_seq, state["a0"].expand(-1, self.steps, -1), state["omega0"].expand(-1, self.steps, -1)], dim=-1)
        return pos, feat, {
            "vel": vel,
            "acc": acc_seq,
            "omega": omega_seq,
            "theta": theta_seq,
            "j": jerk_seq,
            "beta": beta_seq,
        }

    def changing_accel_const_turn(self, state, raw):
        tt, tt2, tt3 = self.time(state["pos"])
        jerk_seq = 1.5 * torch.tanh(raw[..., 0:1])
        omega_seq = state["omega0"] + 0.6 * torch.tanh(raw[..., 1:2])
        s_t = state["speed"] * tt + 0.5 * state["a0"] * tt2 + (1.0 / 6.0) * jerk_seq * tt3
        theta_eff = state["theta0"] + 0.5 * omega_seq * tt
        pos = state["pos"] + s_t * torch.cat([torch.cos(theta_eff), torch.sin(theta_eff)], dim=-1)
        v_next = (state["speed"] + state["a0"] * tt + 0.5 * jerk_seq * tt2).clamp_min(0.001)
        theta_seq = state["theta0"] + omega_seq * tt
        vel = torch.cat([v_next * torch.cos(theta_seq), v_next * torch.sin(theta_seq)], dim=-1)
        acc_seq = state["a0"] + jerk_seq * tt
        feat = torch.cat([jerk_seq, omega_seq, state["a0"].expand(-1, self.steps, -1), state["speed"].expand(-1, self.steps, -1)], dim=-1)
        return pos, feat, {
            "vel": vel,
            "acc": acc_seq,
            "omega": omega_seq,
            "theta": theta_seq,
            "j": jerk_seq,
        }

    def _run_expert(self, name, state, raw):
        if name == "cv":
            return self.constant_velocity(state, raw)
        if name == "ca":
            return self.constant_acceleration(state, raw)
        if name == "ctra":
            return self.accel_turn(state, raw)
        if name == "jerk":
            return self.changing_acceleration(state, raw)
        if name == "jerk_beta":
            return self.changing_accel_turn(state, raw)
        if name == "jerk_turn":
            return self.changing_accel_const_turn(state, raw)
        raise ValueError(f"unsupported structured expert: {name}")

    def forward(self, h, h_global, z, feat, anchor, scale, history_mask=None):
        if z.ndim == 2:
            z = z.unsqueeze(1).expand(-1, self.experts, -1)
        if z.ndim != 3 or z.shape[1] != self.experts:
            raise ValueError("bad latent shape")
        state = self.root_state(feat, anchor, scale, history_mask=history_mask)
        coeff_basis, raw_params = self._predict_coefficients(h_global, z)
        y_all = []
        phys_feat = []
        expert_meta = []
        raw_by_name = {}
        for expert_index, expert_name in enumerate(self.expert_names):
            raw = raw_params[expert_index]
            y_pred, feat_pred, physics = self._run_expert(expert_name, state, raw)
            y_all.append(y_pred)
            phys_feat.append(feat_pred)
            raw_by_name[expert_name] = raw
            expert_meta.append(
                {
                    "name": expert_name,
                    "latent_mode": self._latent_mode(),
                    "coeff_basis": torch.nan_to_num(coeff_basis[expert_index]),
                    "raw_coeff": torch.nan_to_num(raw),
                    "physics": {key: torch.nan_to_num(value) for key, value in physics.items()},
                }
            )
        coeff_scales = self._coeff_scales(raw_by_name)
        for meta in expert_meta:
            meta["coeff_scales"] = coeff_scales
        y_phys = torch.stack(y_all, dim=1)
        phys_feat_tensor = torch.stack(phys_feat, dim=1)

        if self.refiner is not None:
            # theta: [B, K, T, C]
            # Need to pad experts with C=1 (jerk) to C=2
            theta_list = []
            for r in raw_params:
                if r.shape[-1] == 1:
                    theta_list.append(F.pad(r, (0, 1)))
                else:
                    theta_list.append(r)
            theta_stacked = torch.stack(theta_list, dim=1)
            y_phys = self.refiner(theta_stacked, y_phys, z, self.refinement_lambda)

        return {
            "y_all": torch.nan_to_num(y_phys),
            "y_phys": torch.nan_to_num(y_phys),
            "phys_feat": torch.nan_to_num(phys_feat_tensor),
            "expert_aux": expert_meta,
            "expert_names": list(self.expert_names),
        }


class HomogeneousTrajectoryDecoder(nn.Module):
    def __init__(self, hdim, lat=128, dim=128, steps=60, experts=6, shared_head=False):
        super().__init__()
        self.steps = int(steps)
        self.experts = int(experts)
        self.shared_head = bool(shared_head)
        if self.experts < 1:
            raise ValueError("HomogeneousTrajectoryDecoder requires at least one expert")
        inp = hdim + lat
        hid = dim * 2
        if self.shared_head:
            self.head = Mlp(inp, self.steps * 2, hid)
        else:
            self.heads = nn.ModuleList([Mlp(inp, self.steps * 2, hid) for _ in range(self.experts)])

    def forward(self, h, h_global, z, feat, anchor, scale, history_mask=None):
        if z.ndim == 2:
            z = z.unsqueeze(1).expand(-1, self.experts, -1)
        if z.ndim != 3 or z.shape[1] != self.experts:
            raise ValueError("bad latent shape")
        proj_dtype = self.head.net[0].weight.dtype if self.shared_head else self.heads[0].net[0].weight.dtype
        h_global_exp = h_global.to(dtype=proj_dtype).unsqueeze(1).expand(-1, self.experts, -1)
        hz = torch.cat([h_global_exp, z.to(dtype=proj_dtype)], dim=-1)
        outputs = []
        expert_meta = []
        for expert_index in range(self.experts):
            if self.shared_head:
                flat = self.head(hz[:, expert_index])
            else:
                flat = self.heads[expert_index](hz[:, expert_index])
            outputs.append(flat.view(flat.shape[0], self.steps, 2))
            expert_meta.append({"name": f"hom_head_{expert_index + 1}", "physics": {}})
        y_pred = torch.stack(outputs, dim=1)
        return {
            "y_all": torch.nan_to_num(y_pred),
            "y_phys": torch.nan_to_num(y_pred),
            "phys_feat": torch.nan_to_num(y_pred),
            "expert_aux": expert_meta,
            "expert_names": [item["name"] for item in expert_meta],
        }


class HomogeneousTransformerTrajectoryDecoder(nn.Module):
    def __init__(
        self,
        hdim,
        lat=128,
        dim=128,
        steps=60,
        experts=6,
        use_structured_latent=True,
        latent_group_dim=16,
        decoder_layers=3,
        decoder_heads=4,
    ):
        super().__init__()
        self.steps = int(steps)
        self.experts = int(experts)
        self.lat = int(lat)
        self.dim = int(dim)
        self.use_structured_latent = bool(use_structured_latent and lat % 4 == 0)
        self.latent_group_dim = int(latent_group_dim)
        self.decoder_layers = max(int(decoder_layers), 1)
        self.decoder_heads = int(decoder_heads)
        if self.experts < 1:
            raise ValueError("HomogeneousTransformerTrajectoryDecoder requires at least one expert")
        hid = dim * 2
        self.scene_proj = nn.Linear(hdim, dim)
        self.expert_embed = nn.Embedding(self.experts, dim)
        self.latent_add_proj = nn.Linear(lat, dim)
        self.latent_gate_proj = nn.Linear(lat, dim)
        if self.use_structured_latent:
            group_width = lat // 4
            self.latent_group_proj = nn.ModuleList([nn.Linear(group_width, dim) for _ in range(4)])
        else:
            self.latent_group_proj = None
        self.token_blocks = nn.ModuleList([Block(dim, self.decoder_heads, hid * 2) for _ in range(self.decoder_layers)])
        self.token_norm = nn.LayerNorm(dim)
        self.traj_heads = nn.ModuleList([nn.Linear(dim, self.steps * 2) for _ in range(self.experts)])

    def _structured_latent_features(self, z):
        if not self.use_structured_latent:
            return z.new_zeros((z.shape[0], z.shape[1], self.dim))
        groups = torch.chunk(z, 4, dim=-1)
        parts = [proj(group) for proj, group in zip(self.latent_group_proj, groups)]
        return torch.stack(parts, dim=0).sum(dim=0)

    def _build_tokens(self, h_global, z):
        proj_dtype = self.scene_proj.weight.dtype
        h_global = h_global.to(dtype=proj_dtype)
        z = z.to(dtype=proj_dtype)
        scene_token = self.scene_proj(h_global).unsqueeze(1)
        latent_add = self.latent_add_proj(z) + self._structured_latent_features(z)
        latent_gate = torch.tanh(self.latent_gate_proj(z))
        expert_ids = torch.arange(self.experts, device=z.device, dtype=torch.long)
        expert_token = self.expert_embed(expert_ids).unsqueeze(0)
        tokens = scene_token * (1.0 + 0.5 * latent_gate) + latent_add + expert_token
        for block in self.token_blocks:
            tokens = block(tokens)
        return self.token_norm(tokens)

    def forward(self, h, h_global, z, feat, anchor, scale, history_mask=None):
        if z.ndim == 2:
            z = z.unsqueeze(1).expand(-1, self.experts, -1)
        if z.ndim != 3 or z.shape[1] != self.experts:
            raise ValueError("bad latent shape")
        tokens = self._build_tokens(h_global, z)
        outputs = []
        expert_meta = []
        for expert_index, head in enumerate(self.traj_heads):
            flat = head(tokens[:, expert_index])
            outputs.append(flat.view(flat.shape[0], self.steps, 2))
            expert_meta.append({"name": f"tf_head_{expert_index + 1}", "physics": {}})
        y_pred = torch.stack(outputs, dim=1)
        return {
            "y_all": torch.nan_to_num(y_pred),
            "y_phys": torch.nan_to_num(y_pred),
            "phys_feat": torch.nan_to_num(y_pred),
            "expert_aux": expert_meta,
            "expert_names": [item["name"] for item in expert_meta],
        }
