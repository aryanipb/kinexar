from kinexar.models.cvae.moe_decoder import HomogeneousTrajectoryDecoder
from kinexar.models.cvae.moe_decoder import HomogeneousTransformerTrajectoryDecoder
from kinexar.models.cvae.moe_decoder import MoEPhysicsDecoder


def build_decoder(
    decoder_family,
    dim,
    lat,
    future_steps,
    expert_names,
    expert_count,
    shared_decoder,
    use_transformer_coeff_decoder=True,
    use_mlp_coeff_decoder=False,
    use_structured_latent=True,
    latent_group_dim=16,
    decoder_layers=3,
    decoder_heads=4,
    use_residual_refinement=False,
    refinement_lambda=0.35,
    ):
    if decoder_family == "structured_bank":
        return MoEPhysicsDecoder(
            hdim=dim,
            lat=lat,
            dim=dim,
            steps=future_steps,
            expert_count=expert_count,
            expert_names=expert_names,
            use_transformer_coeff_decoder=use_transformer_coeff_decoder,
            use_mlp_coeff_decoder=use_mlp_coeff_decoder,
            use_structured_latent=use_structured_latent,
            latent_group_dim=latent_group_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            use_residual_refinement=use_residual_refinement,
            refinement_lambda=refinement_lambda,
        )

    if decoder_family == "homogeneous_mlp":
        return HomogeneousTrajectoryDecoder(
            hdim=dim,
            lat=lat,
            dim=dim,
            steps=future_steps,
            experts=int(expert_count),
            shared_head=bool(shared_decoder),
        )
    if decoder_family == "homogeneous_transformer":
        return HomogeneousTransformerTrajectoryDecoder(
            hdim=dim,
            lat=lat,
            dim=dim,
            steps=future_steps,
            experts=int(expert_count),
            use_structured_latent=bool(use_structured_latent),
            latent_group_dim=int(latent_group_dim),
            decoder_layers=int(decoder_layers),
            decoder_heads=int(decoder_heads),
        )
    raise ValueError(f"unknown decoder_family: {decoder_family}")
