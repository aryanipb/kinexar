import torch
import torch.nn.functional as F


KL_ANNEAL_STEPS = 15000
KL_FREE_BITS = 0.10
EXPERT_WEIGHT = 0.5
VEL_WEIGHT = 2.0
LATENT_GROUP_NAMES = ("route", "speed", "interaction", "style")


class KLAnnealer:
    def __init__(self, total_steps):
        self.total_steps = max(int(total_steps), 1)

    def __call__(self, step):
        if step is None:
            return 1.0
        return min(1.0, float(step) / float(self.total_steps))


def kld_gaussian(post_mu, post_logvar, prior_mu, prior_logvar):
    post_logvar = torch.clamp(post_logvar, -10.0, 10.0)
    prior_logvar = torch.clamp(prior_logvar, -10.0, 10.0)
    post_var = torch.exp(post_logvar)
    prior_var = torch.exp(prior_logvar)
    term = prior_logvar - post_logvar + (post_var + (post_mu - prior_mu).pow(2)) / prior_var - 1.0
    return 0.5 * term.mean(dim=-1)


def kld_gaussian_sum(post_mu, post_logvar, prior_mu, prior_logvar):
    post_logvar = torch.clamp(post_logvar, -10.0, 10.0)
    prior_logvar = torch.clamp(prior_logvar, -10.0, 10.0)
    post_var = torch.exp(post_logvar)
    prior_var = torch.exp(prior_logvar)
    term = prior_logvar - post_logvar + (post_var + (post_mu - prior_mu).pow(2)) / prior_var - 1.0
    return 0.5 * term.sum(dim=-1)


def kl_anneal_weight(global_step, anneal_steps=KL_ANNEAL_STEPS):
    return KLAnnealer(anneal_steps)(global_step)


def _pos_loss(pred, gt):
    return ((pred - gt) ** 2).sum(dim=-1).mean(dim=-1)


def _wrap_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def expert1_loss(pred, gt, aux):
    vel = aux["vel"]
    fit = _pos_loss(pred, gt)
    physics = ((vel[:, 1:] - vel[:, :-1]) ** 2).sum(dim=-1).mean(dim=-1)
    return {"loss": fit + physics, "fit": fit, "physics": physics}


def expert2_loss(pred, gt, aux):
    acc = aux["acc"]
    fit = _pos_loss(pred, gt)
    physics = ((acc[:, 1:] - acc[:, :-1]) ** 2).sum(dim=-1).mean(dim=-1)
    return {"loss": fit + physics, "fit": fit, "physics": physics}


def expert3_loss(pred, gt, aux, dt):
    omega = aux["omega"]
    theta = aux["theta"]
    theta0 = theta[:, :1]
    theta_prev = torch.cat([theta0, theta[:, :-1]], dim=1)
    theta_step = _wrap_angle(theta - theta_prev - omega * dt)
    fit = _pos_loss(pred, gt)
    omega_const = ((omega[:, 1:] - omega[:, :-1]) ** 2).mean(dim=(-1, -2))
    theta_dyn = (theta_step ** 2).mean(dim=(-1, -2))
    physics = omega_const + theta_dyn
    return {"loss": fit + physics, "fit": fit, "physics": physics}


def expert4_loss(pred, gt, aux):
    acc = aux["acc"]
    fit = _pos_loss(pred, gt)
    physics = ((acc[:, 1:] - acc[:, :-1]) ** 2).mean(dim=(-1, -2))
    return {"loss": fit + physics, "fit": fit, "physics": physics}


def expert5_loss(pred, gt, aux):
    fit = _pos_loss(pred, gt)
    jerk_pen = (aux["j"] ** 2).mean(dim=(-1, -2))
    beta_pen = (aux["beta"] ** 2).mean(dim=(-1, -2))
    physics = jerk_pen + beta_pen
    return {"loss": fit + physics, "fit": fit, "physics": physics}


def expert6_loss(pred, gt, aux):
    fit = _pos_loss(pred, gt)
    jerk_pen = (aux["j"] ** 2).mean(dim=(-1, -2))
    omega_smooth = ((aux["omega"][:, 1:] - aux["omega"][:, :-1]) ** 2).mean(dim=(-1, -2))
    physics = jerk_pen + omega_smooth
    return {"loss": fit + physics, "fit": fit, "physics": physics}


def direct_head_loss(pred, gt):
    fit = _pos_loss(pred, gt)
    physics = fit.new_zeros(fit.shape)
    return {"loss": fit, "fit": fit, "physics": physics}


def _summary_errors(pred, gt):
    # BASE_SCALE=50.0 converts normalized 'Silicon' units back to real meters
    BASE_SCALE = 50.0
    err = torch.linalg.vector_norm((pred * BASE_SCALE) - (gt[:, None] * BASE_SCALE), dim=-1)
    ade_all = err.mean(dim=-1)
    fde_all = err[..., -1]
    return ade_all, fde_all


def velocity_loss(pred, gt):
    if pred.ndim != 4 or gt.ndim != 3:
        raise ValueError("bad velocity loss shapes")
    pred_vel = pred[:, :, 1:] - pred[:, :, :-1]
    gt_vel = gt[:, 1:] - gt[:, :-1]
    return (pred_vel - gt_vel[:, None]).abs().mean()


def _time_weights_like(seq, mode="quadratic", strength=2.5):
    length = seq.shape[-2]
    if length < 1:
        raise ValueError("bad time length")
    if length == 1:
        shape = [1] * seq.ndim
        shape[-2] = 1
        return torch.ones(shape, device=seq.device, dtype=seq.dtype)
    idx = torch.arange(length, device=seq.device, dtype=seq.dtype)
    prog = idx / max(length - 1, 1)
    if mode == "uniform":
        w = torch.ones_like(prog)
    elif mode == "quadratic":
        w = 1.0 + float(strength) * prog.pow(2.0)
    elif mode == "endpoint_heavy":
        w = 1.0 + float(strength) * prog.pow(4.0)
    else:
        raise ValueError(f"bad time weight mode: {mode}")
    shape = [1] * seq.ndim
    shape[-2] = length
    shape[-1] = 1
    return w.view(*shape)


def _weighted_l1(pred, gt, weight):
    return ((pred - gt).abs() * weight).mean()


def _weighted_smooth_l1(pred, gt, weight):
    loss = F.smooth_l1_loss(pred, gt, reduction="none")
    return (loss * weight).mean()


def _delta(seq):
    if seq.shape[-2] < 2:
        shape = list(seq.shape)
        shape[-2] = 0
        return seq.new_zeros(shape)
    return seq[..., 1:, :] - seq[..., :-1, :]


def _final_direction(seq):
    if seq.shape[-2] < 2:
        return None
    delta = seq[..., -1, :] - seq[..., -2, :]
    return delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _curvature(seq):
    vel = _delta(seq)
    if vel.shape[-2] < 2:
        shape = list(seq.shape)
        shape[-2] = 0
        return seq.new_zeros(shape)
    unit = vel / vel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return unit[..., 1:, :] - unit[..., :-1, :]


def _multi_step_loss(pred, gt):
    if pred is None or gt is None:
        return None
    if pred.shape != gt.shape:
        raise ValueError("bad multi-step shapes")
    valid = gt.abs().sum(dim=-1, keepdim=True) > 0
    return ((pred - gt).abs() * valid.to(pred.dtype)).sum() / valid.to(pred.dtype).sum().clamp_min(1.0)


def _structured_kl_from_flat(post_mu, post_logvar, prior_mu, prior_logvar, free_bits):
    if post_mu.shape[-1] % 4 != 0:
        raise ValueError("structured latent dimension must be divisible by 4")
    group_dim = post_mu.shape[-1] // 4
    group_terms = {}
    raw_terms = {}
    for group_index, group_name in enumerate(LATENT_GROUP_NAMES):
        start = group_index * group_dim
        end = start + group_dim
        raw = kld_gaussian_sum(post_mu[:, start:end], post_logvar[:, start:end], prior_mu[:, start:end], prior_logvar[:, start:end])
        raw_terms[group_name] = raw.mean()
        group_terms[group_name] = raw.clamp_min(float(free_bits)).mean()
    return raw_terms, group_terms


def _select_single_mode(traj, score_tensor=None, gt=None):
    if traj is None:
        return None
    if traj.ndim != 4:
        raise ValueError("bad trajectory shape")
    batch_idx = torch.arange(traj.shape[0], device=traj.device)
    if score_tensor is not None:
        mode_idx = score_tensor.argmax(dim=-1)
    elif gt is not None:
        ade_all, _ = _summary_errors(traj, gt)
        mode_idx = ade_all.argmin(dim=-1)
    else:
        mode_idx = torch.zeros((traj.shape[0],), dtype=torch.long, device=traj.device)
    return traj[batch_idx, mode_idx]


def _route_channel(traj):
    disp = traj[:, -1] - traj[:, 0]
    return torch.atan2(disp[:, 1], disp[:, 0])


def _speed_channel(traj):
    step = traj[:, 1:] - traj[:, :-1]
    return torch.linalg.vector_norm(step, dim=-1).mean(dim=-1)


def _interaction_channel(traj, gt, radius=2.0):
    gt_final = gt[:, -1]
    dist = torch.linalg.vector_norm(traj - gt_final.unsqueeze(1), dim=-1)
    inside = dist <= float(radius)
    steps = traj.shape[1]
    idx = torch.arange(steps, device=traj.device, dtype=traj.dtype).view(1, steps)
    fill = torch.full_like(idx, float(steps))
    return torch.where(inside, idx, fill).amin(dim=1)


def _style_channel(traj):
    if traj.shape[1] < 3:
        return traj.new_zeros((traj.shape[0],))
    second = traj[:, 2:] - 2.0 * traj[:, 1:-1] + traj[:, :-2]
    return torch.linalg.vector_norm(second, dim=-1).mean(dim=-1)


def _channel_matrix(selected_base, selected_variants, gt):
    base_channels = {
        "route": _route_channel(selected_base),
        "speed": _speed_channel(selected_base),
        "interaction": _interaction_channel(selected_base, gt),
        "style": _style_channel(selected_base),
    }
    deltas = {}
    factor_order = ("route", "speed", "interaction", "style")
    for factor_name in factor_order:
        cf = selected_variants[factor_name]
        cf_channels = {
            "route": _route_channel(cf),
            "speed": _speed_channel(cf),
            "interaction": _interaction_channel(cf, gt),
            "style": _style_channel(cf),
        }
        for channel_name in factor_order:
            deltas[(factor_name, channel_name)] = (cf_channels[channel_name] - base_channels[channel_name]).abs()
    return deltas


def moe(
    pred,
    gt,
    expert_aux=None,
    post_mu=None,
    post_logvar=None,
    prior_mu=None,
    prior_logvar=None,
    kl_weight=None,
    global_step=None,
    free_bits=KL_FREE_BITS,
    dt=0.1,
    use_physics_loss=True,
    warmup_steps=KL_ANNEAL_STEPS,
    cfcp_base_abs=None,
    cfcp_route_abs=None,
    cfcp_speed_abs=None,
    cfcp_interaction_abs=None,
    cfcp_style_abs=None,
    cfcp_active=False,
    cfcp_sigma=0.15,
    cfcp_tau=0.25,
    lambda_ownership=0.10,
    lambda_leakage=0.05,
    lambda_stability=0.02,
    use_cfcp_ownership=True,
    use_cfcp_leakage=True,
    use_cfcp_stability=True,
    compute_cri_metrics=True,
):
    if pred.ndim != 4:
        raise ValueError("expected batched expert predictions")
    if expert_aux is None:
        raise ValueError("missing expert_aux")
    if isinstance(expert_aux, dict):
        legacy_names = ("cv", "ca", "ctra", "jerk", "jerk_beta", "jerk_turn")
        normalized = []
        for name in legacy_names:
            if name in expert_aux:
                normalized.append({"name": name, "physics": expert_aux[name]})
        expert_aux = normalized
    if not isinstance(expert_aux, list) or len(expert_aux) != pred.shape[1]:
        raise ValueError("expert_aux must describe each expert prediction")
    if expert_aux and isinstance(expert_aux[0], dict):
        cfcp_shared = expert_aux[0].get("cfcp_shared", {})
        if cfcp_base_abs is None:
            cfcp_base_abs = cfcp_shared.get("base_abs")
        if cfcp_route_abs is None:
            cfcp_route_abs = cfcp_shared.get("route_abs")
        if cfcp_speed_abs is None:
            cfcp_speed_abs = cfcp_shared.get("speed_abs")
        if cfcp_interaction_abs is None:
            cfcp_interaction_abs = cfcp_shared.get("interaction_abs")
        if cfcp_style_abs is None:
            cfcp_style_abs = cfcp_shared.get("style_abs")
        if not cfcp_active:
            cfcp_active = bool(cfcp_shared.get("active", False))
        if cfcp_sigma == 0.15:
            cfcp_sigma = float(cfcp_shared.get("sigma", cfcp_sigma))
    packs = []
    for expert_index, meta in enumerate(expert_aux):
        name = str(meta.get("name", f"expert_{expert_index + 1}"))
        physics = meta.get("physics", {})
        if (not use_physics_loss) or name.startswith("hom_head") or name == "shared_head":
            pack = direct_head_loss(pred[:, expert_index], gt)
        elif name == "cv":
            pack = expert1_loss(pred[:, expert_index], gt, physics)
        elif name == "ca":
            pack = expert2_loss(pred[:, expert_index], gt, physics)
        elif name == "ctra":
            pack = expert3_loss(pred[:, expert_index], gt, physics, dt=dt)
        elif name == "jerk":
            pack = expert4_loss(pred[:, expert_index], gt, physics)
        elif name == "jerk_beta":
            pack = expert5_loss(pred[:, expert_index], gt, physics)
        elif name == "jerk_turn":
            pack = expert6_loss(pred[:, expert_index], gt, physics)
        else:
            pack = direct_head_loss(pred[:, expert_index], gt)
        packs.append(pack)
    expert_losses = [pack["loss"].mean() for pack in packs]
    fit_terms = [pack["fit"].mean() for pack in packs]
    physics_terms = [pack["physics"].mean() for pack in packs]
    vel = velocity_loss(pred, gt)
    recon = pred.new_zeros(())
    for expert_loss in expert_losses:
        recon = recon + EXPERT_WEIGHT * expert_loss
    recon = recon + VEL_WEIGHT * vel
    raw_kl = pred.new_zeros(())
    kl_total = pred.new_zeros(())
    kl_loss = pred.new_zeros(())
    kl_route = pred.new_zeros(())
    kl_speed = pred.new_zeros(())
    kl_interaction = pred.new_zeros(())
    kl_style = pred.new_zeros(())
    cur_kl_weight = pred.new_zeros(())
    structured_latent = bool(expert_aux[0].get("latent_mode") == "structured") if expert_aux else False
    if post_mu is not None and post_logvar is not None and prior_mu is not None and prior_logvar is not None:
        if structured_latent:
            raw_groups, kl_groups = _structured_kl_from_flat(post_mu, post_logvar, prior_mu, prior_logvar, free_bits=free_bits)
            raw_kl = sum(raw_groups.values())
            kl_route = kl_groups["route"]
            kl_speed = kl_groups["speed"]
            kl_interaction = kl_groups["interaction"]
            kl_style = kl_groups["style"]
            kl_total = kl_route + kl_speed + kl_interaction + kl_style
        else:
            raw_kl_per_sample = kld_gaussian(post_mu, post_logvar, prior_mu, prior_logvar)
            raw_kl = raw_kl_per_sample.mean()
            kl_total = torch.clamp(raw_kl_per_sample, min=float(free_bits)).mean()
        if global_step is not None:
            cur_kl_weight = torch.as_tensor(
                kl_anneal_weight(global_step, anneal_steps=warmup_steps),
                device=pred.device,
                dtype=pred.dtype,
            )
        elif kl_weight is not None:
            cur_kl_weight = torch.as_tensor(kl_weight, device=pred.device, dtype=pred.dtype)
        else:
            cur_kl_weight = pred.new_ones(())
        kl_loss = cur_kl_weight * kl_total
    score_tensors = []
    for meta in expert_aux:
        score = meta.get("score")
        if isinstance(score, torch.Tensor):
            score_tensors.append(score)
    score_tensor = torch.stack(score_tensors, dim=1) if len(score_tensors) == pred.shape[1] else None
    mode_ce = pred.new_zeros(())
    mode_entropy = pred.new_zeros(())
    ade_all, fde_all = _summary_errors(pred, gt)
    if score_tensor is not None:
        probs = torch.softmax(score_tensor, dim=-1)
        best_mode_index = ade_all.argmin(dim=-1)
        mode_ce = F.cross_entropy(score_tensor, best_mode_index)
        mode_entropy = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    loss_ownership = pred.new_zeros(())
    loss_leakage = pred.new_zeros(())
    loss_stability = pred.new_zeros(())
    cri = pred.new_zeros(())
    delta_logs = {
        (factor_name, channel_name): pred.new_zeros(())
        for factor_name in LATENT_GROUP_NAMES
        for channel_name in LATENT_GROUP_NAMES
    }
    if bool(cfcp_active) and all(x is not None for x in (cfcp_base_abs, cfcp_route_abs, cfcp_speed_abs, cfcp_interaction_abs, cfcp_style_abs)):
        selected_base = _select_single_mode(cfcp_base_abs, score_tensor=score_tensor, gt=gt)
        selected_variants = {
            "route": _select_single_mode(cfcp_route_abs, score_tensor=score_tensor, gt=gt),
            "speed": _select_single_mode(cfcp_speed_abs, score_tensor=score_tensor, gt=gt),
            "interaction": _select_single_mode(cfcp_interaction_abs, score_tensor=score_tensor, gt=gt),
            "style": _select_single_mode(cfcp_style_abs, score_tensor=score_tensor, gt=gt),
        }
        delta_matrix = _channel_matrix(selected_base, selected_variants, gt)
        for key, value in delta_matrix.items():
            delta_logs[key] = value.mean()
        ownership_terms = []
        off_diag_terms = []
        diagonal_terms = []
        for factor_name in LATENT_GROUP_NAMES:
            logits = torch.stack([delta_matrix[(factor_name, channel_name)] / float(cfcp_tau) for channel_name in LATENT_GROUP_NAMES], dim=-1)
            target = torch.full((logits.shape[0],), LATENT_GROUP_NAMES.index(factor_name), dtype=torch.long, device=logits.device)
            ownership_terms.append(F.cross_entropy(logits, target))
            diagonal_terms.append(delta_matrix[(factor_name, factor_name)])
            for channel_name in LATENT_GROUP_NAMES:
                if channel_name != factor_name:
                    off_diag_terms.append(delta_matrix[(factor_name, channel_name)].mean())
        diagonal_stack = torch.stack(diagonal_terms, dim=-1)
        if bool(use_cfcp_ownership):
            loss_ownership = torch.stack(ownership_terms).mean()
        if bool(use_cfcp_leakage):
            loss_leakage = torch.stack(off_diag_terms).mean()
        if bool(use_cfcp_stability):
            loss_stability = diagonal_stack.std(dim=0).mean()
        if bool(compute_cri_metrics):
            cri = torch.stack(off_diag_terms).mean() if off_diag_terms else pred.new_zeros(())
    loss = recon + kl_loss + 0.1 * mode_ce
    if bool(use_cfcp_ownership):
        loss = loss + float(lambda_ownership) * loss_ownership
    if bool(use_cfcp_leakage):
        loss = loss + float(lambda_leakage) * loss_leakage
    if bool(use_cfcp_stability):
        loss = loss + float(lambda_stability) * loss_stability
    prior_mu_norm = prior_mu.norm(dim=-1).mean() if prior_mu is not None else pred.new_zeros(())
    post_mu_norm = post_mu.norm(dim=-1).mean() if post_mu is not None else pred.new_zeros(())
    if post_logvar is not None:
        latent_std_mean = torch.exp(0.5 * post_logvar).mean()
    elif prior_logvar is not None:
        latent_std_mean = torch.exp(0.5 * prior_logvar).mean()
    else:
        latent_std_mean = pred.new_zeros(())
    out = {
        "loss": loss,
        "recon": recon,
        "fit_recon": sum(fit_terms) / len(fit_terms),
        "kl": kl_loss,
        "raw_kl": raw_kl,
        "effective_kl": kl_loss,
        "kl_weight": cur_kl_weight,
        "beta": cur_kl_weight,
        "beta_kl": cur_kl_weight,
        "vel_loss": vel,
        "expert_losses": torch.stack(expert_losses),
        "expert_physics": torch.stack(physics_terms),
        "expert_ade": ade_all,
        "expert_fde": fde_all,
        "mode_ce": mode_ce,
        "mode_entropy": mode_entropy,
        "kl_route": kl_route,
        "kl_speed": kl_speed,
        "kl_interaction": kl_interaction,
        "kl_style": kl_style,
        "kl_total": kl_total,
        "prior_mu_norm": prior_mu_norm,
        "post_mu_norm": post_mu_norm,
        "latent_std_mean": latent_std_mean,
    }
    latent_std_route = pred.new_zeros(())
    latent_std_speed = pred.new_zeros(())
    latent_std_interaction = pred.new_zeros(())
    latent_std_style = pred.new_zeros(())
    if structured_latent:
        source_logvar = post_logvar if post_logvar is not None else prior_logvar
        if source_logvar is not None:
            source_logvar = torch.clamp(source_logvar, -10.0, 10.0)
            group_dim = source_logvar.shape[-1] // 4
            latent_std_route = torch.exp(0.5 * source_logvar[:, 0 * group_dim : 1 * group_dim]).mean()
            latent_std_speed = torch.exp(0.5 * source_logvar[:, 1 * group_dim : 2 * group_dim]).mean()
            latent_std_interaction = torch.exp(0.5 * source_logvar[:, 2 * group_dim : 3 * group_dim]).mean()
            latent_std_style = torch.exp(0.5 * source_logvar[:, 3 * group_dim : 4 * group_dim]).mean()
    out["latent_std_route"] = latent_std_route
    out["latent_std_speed"] = latent_std_speed
    out["latent_std_interaction"] = latent_std_interaction
    out["latent_std_style"] = latent_std_style
    if expert_aux and isinstance(expert_aux[0], dict):
        coeff_scales = expert_aux[0].get("coeff_scales", {})
        out["v_scale"] = coeff_scales.get("v_scale", pred.new_zeros(()))
        out["a_scale"] = coeff_scales.get("a_scale", pred.new_zeros(()))
        out["omega_scale"] = coeff_scales.get("omega_scale", pred.new_zeros(()))
        out["j_scale"] = coeff_scales.get("j_scale", pred.new_zeros(()))
        out["alpha_scale"] = coeff_scales.get("alpha_scale", pred.new_zeros(()))
    else:
        out["v_scale"] = pred.new_zeros(())
        out["a_scale"] = pred.new_zeros(())
        out["omega_scale"] = pred.new_zeros(())
        out["j_scale"] = pred.new_zeros(())
        out["alpha_scale"] = pred.new_zeros(())
    out["loss_ownership"] = loss_ownership
    out["loss_leakage"] = loss_leakage
    out["loss_stability"] = loss_stability
    out["cri"] = cri
    for factor_name in LATENT_GROUP_NAMES:
        for channel_name in LATENT_GROUP_NAMES:
            out[f"delta_{factor_name}_{channel_name}"] = delta_logs[(factor_name, channel_name)]
    report_count = max(6, len(expert_losses))
    zero = pred.new_zeros(())
    for idx in range(report_count):
        out[f"expert{idx + 1}_loss"] = expert_losses[idx] if idx < len(expert_losses) else zero
        out[f"expert{idx + 1}_fit"] = fit_terms[idx] if idx < len(fit_terms) else zero
        out[f"expert{idx + 1}_physics"] = physics_terms[idx] if idx < len(physics_terms) else zero
    return out
