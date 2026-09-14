import torch

from kinexar.models.trajectory_transforms import build_latent_generator
from kinexar.models.trajectory_transforms import sample_latent_tensor
from kinexar.models.trajectory_transforms import to_local_coordinates
from kinexar.models.trajectory_types import LatentState


def sample_latent_state(model, batch, ctx, mode):
    latent_count = model.cvae_dec.experts

    if not model.use_cvae_latent:
        future_local = batch.future
        z_expert = torch.zeros(
            (ctx.h.shape[0], latent_count, model.lat),
            device=batch.feat.device,
            dtype=ctx.h.dtype,
        )
        return LatentState(
            future_local=future_local,
            prior_mu=None,
            prior_logvar=None,
            post_mu=None,
            post_logvar=None,
            z_expert=z_expert,
        )

    prior_pack = model.prior(ctx.h_global, ctx.ctx_mean)
    prior_mu = prior_pack["mu"]
    prior_logvar = prior_pack["logvar"]

    if mode == "train":
        if batch.future is None:
            raise ValueError("missing future")
        # Pre-processed future is already in the focal-relative, scaled manifold
        future_local = batch.future
        # model.cvae_enc builds the posterior latent from context plus target future.
        h_cvae = ctx.h_nodes.reshape(ctx.h_nodes.shape[0], -1, ctx.h_nodes.shape[-1])
        mask_cvae = ctx.history_mask.reshape(ctx.history_mask.shape[0], -1)

        posterior_pack = model.cvae_enc(h_cvae, future_local, sample=True, context_mask=mask_cvae)
        post_mu = posterior_pack["mu"]
        post_logvar = posterior_pack["logvar"]
        # sample_latent_tensor draws one latent per expert branch.
        z_expert = sample_latent_tensor(
            model.lat,
            ctx.h.shape[0],
            latent_count,
            batch.feat.device,
            ctx.h.dtype,
            mu=post_mu,
            logvar=post_logvar,
        )
        return LatentState(
            future_local=future_local,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            post_mu=post_mu,
            post_logvar=post_logvar,
            z_expert=z_expert,
        )

    # build_latent_generator keeps eval sampling deterministic across runs.
    generator = build_latent_generator(model.eval_latent_seed, batch.feat.device)
    # sample_latent_tensor draws prior samples for inference-time diversity.
    z_expert = sample_latent_tensor(
        model.lat,
        ctx.h.shape[0],
        latent_count,
        batch.feat.device,
        ctx.h.dtype,
        mu=prior_mu,
        logvar=prior_logvar,
        generator=generator,
    )
    return LatentState(
        future_local=None,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        post_mu=None,
        post_logvar=None,
        z_expert=z_expert,
    )
