from kinexar.models.trajectory_transforms import repeat_batch_rows
from kinexar.models.trajectory_transforms import to_abs_coordinates
from kinexar.models.trajectory_transforms import to_local_coordinates


def decode_predictions(model, batch, ctx, latent):
    # model.cvae_dec turns context and latent codes into trajectory hypotheses.
    decoder_out = model.cvae_dec(
        ctx.h,
        ctx.h_global,
        latent.z_expert,
        batch.feat,
        ctx.anchor,
        batch.scale,
        history_mask=ctx.history_mask,
    )
    # pred_local is already in the focal-relative, scaled (1/50) manifold
    pred_local = decoder_out["y_all"]
    # to_abs_coordinates converts relative scaled predictions back to absolute meters
    pred_abs = to_abs_coordinates(pred_local, ctx.anchor, batch.scale)
    return decoder_out, pred_local, pred_abs


def pack_outputs(batch, ctx, latent, decoder_out, pred_local, pred_abs):
    # In our silicon manifold, pred_local IS the target coordinate space.
    # y_all and y_final_local are used by the loss/metrics and must match the 1/50 scale.
    return {
        "y_all": pred_local,
        "y_raw_abs": pred_abs,
        "y_raw_local": pred_local,
        "pred_local": pred_local,
        "pred_abs": pred_abs,
        "y_abs": pred_abs,
        "y_local": pred_local,
        "y_final_local": pred_local,
        "y_final_abs": pred_abs,
        "y_phys": decoder_out["y_phys"],
        "phys_feat": decoder_out["phys_feat"],
        "expert_aux": decoder_out["expert_aux"],
        "anchor": ctx.anchor,
        "scale": batch.scale,
        "prior_mu": latent.prior_mu,
        "prior_logvar": latent.prior_logvar,
        "post_mu": latent.post_mu,
        "post_logvar": latent.post_logvar,
        "z_expert": latent.z_expert,
    }
