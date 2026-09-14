import torch
from torch import nn

from kinexar.models.cvae.encoder import Encoder
from kinexar.models.cvae.prior import Prior
from kinexar.models.spatial_temporal_encoder import SpatialTemporalEncoder
from kinexar.models.trajectory_context import encode_context
from kinexar.models.trajectory_decode import decode_predictions
from kinexar.models.trajectory_decode import pack_outputs
from kinexar.models.trajectory_decoder_factory import build_decoder
from kinexar.models.trajectory_inputs import normalize_inputs
from kinexar.models.trajectory_latents import sample_latent_state
from kinexar.models.trajectory_onnx import export_trajectory_model_onnx
from kinexar.models.trajectory_transforms import build_latent_generator
from kinexar.models.trajectory_transforms import repeat_batch_rows
from kinexar.models.trajectory_transforms import sample_latent_tensor
from kinexar.models.trajectory_transforms import to_abs_coordinates
from kinexar.models.trajectory_transforms import to_local_coordinates
from kinexar.dataset.preprocessing.graph_builder import build
from kinexar.dataset.preprocessing.graph_builder import DEFAULT_AGENT_RADIUS_METERS


class TrajectoryModel(nn.Module):
    def __init__(
        self,
        din,
        dim=128,
        lat=128,
        future_steps: int = 60,
        decoder_family: str = "structured_bank",
        expert_names: list[str] | None = None,
        expert_count: int = 6,
        use_cvae_latent: bool = True,
        shared_decoder: bool = False,
        encoder_mode: str = "full",
        use_transformer_coeff_decoder: bool = True,
        use_mlp_coeff_decoder: bool = False,
        use_structured_latent: bool = True,
        use_flat_latent: bool = False,
        use_residual_refinement: bool = False,
        refinement_lambda: float = 0.35,
        latent_group_dim: int = 16,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        use_cfcp: bool = True,
        use_cfcp_ownership: bool = True,
        use_cfcp_leakage: bool = True,
        use_cfcp_stability: bool = True,
        cfcp_every_n_steps: int = 2,
        cfcp_sigma: float = 0.15,
    ):
        super().__init__()
        self.lat = lat
        self.eval_latent_seed = 0
        self.future_steps = int(future_steps)
        self.decoder_family = str(decoder_family)
        self.use_cvae_latent = bool(use_cvae_latent)
        # These flags are accepted to stay compatible with the experiment config API.
        self.model_options = {
            "use_transformer_coeff_decoder": bool(use_transformer_coeff_decoder),
            "use_mlp_coeff_decoder": bool(use_mlp_coeff_decoder),
            "use_structured_latent": bool(use_structured_latent),
            "use_flat_latent": bool(use_flat_latent),
            "use_residual_refinement": bool(use_residual_refinement),
            "refinement_lambda": float(refinement_lambda),
            "latent_group_dim": int(latent_group_dim),
            "decoder_layers": int(decoder_layers),
            "decoder_heads": int(decoder_heads),
            "use_cfcp": bool(use_cfcp),
            "use_cfcp_ownership": bool(use_cfcp_ownership),
            "use_cfcp_leakage": bool(use_cfcp_leakage),
            "use_cfcp_stability": bool(use_cfcp_stability),
            "cfcp_every_n_steps": int(cfcp_every_n_steps),
            "cfcp_sigma": float(cfcp_sigma),
        }
        if self.future_steps < 1:
            raise ValueError("future_steps must be positive")
        # SpatialTemporalEncoder converts agent history into node embeddings.
        self.encoder = SpatialTemporalEncoder(din, dim=dim, mode=encoder_mode)
        # Encoder builds the posterior latent from context plus target future.
        self.cvae_enc = Encoder(hdim=dim, lat=lat, dim=dim)
        # Prior predicts the latent distribution from encoded scene context.
        self.prior = Prior(hdim=dim, lat=lat)
        # build_decoder selects the concrete trajectory decoder family.
        self.cvae_dec = build_decoder(
            decoder_family=self.decoder_family,
            dim=dim,
            lat=lat,
            future_steps=self.future_steps,
            expert_names=expert_names,
            expert_count=expert_count,
            shared_decoder=shared_decoder,
            use_transformer_coeff_decoder=self.model_options["use_transformer_coeff_decoder"],
            use_mlp_coeff_decoder=self.model_options["use_mlp_coeff_decoder"],
            use_structured_latent=self.model_options["use_structured_latent"],
            latent_group_dim=self.model_options["latent_group_dim"],
            decoder_layers=self.model_options["decoder_layers"],
            decoder_heads=self.model_options["decoder_heads"],
            use_residual_refinement=self.model_options.get("use_residual_refinement", False),
            refinement_lambda=self.model_options.get("refinement_lambda", 0.35),
        )

    def repeat_map(self, x, k):
        return repeat_batch_rows(x, k)

    def to_abs(self, y_local, anchor, scale):
        return to_abs_coordinates(y_local, anchor, scale)

    def to_local(self, y_abs, anchor, scale):
        return to_local_coordinates(y_abs, anchor, scale)

    def _latent_generator(self, device):
        return build_latent_generator(self.eval_latent_seed, device)

    def prior_sample(self, batch, device, dtype):
        gen = self._latent_generator(device)
        return torch.randn((batch, self.lat), generator=gen, device=device, dtype=dtype)

    def sample_latents(self, batch, count, device, dtype, mu=None, logvar=None, generator=None):
        if generator is None:
            generator = self._latent_generator(device)
        return sample_latent_tensor(self.lat, batch, count, device, dtype, mu=mu, logvar=logvar, generator=generator)

    def _mean_velocity(self, seq):
        if seq.shape[1] < 2:
            return torch.zeros((seq.shape[0], seq.shape[-1]), device=seq.device, dtype=seq.dtype)
        return (seq[:, 1:] - seq[:, :-1]).mean(dim=1)

    def _last_velocity(self, seq):
        if seq.shape[1] < 2:
            return torch.zeros((seq.shape[0], seq.shape[-1]), device=seq.device, dtype=seq.dtype)
        return seq[:, -1] - seq[:, -2]

    def _build_multi_targets(self, seq, k):
        if seq is None or k <= 0:
            return None
        out = seq.new_zeros((seq.shape[0], seq.shape[1], k, seq.shape[2]))
        for step in range(k):
            valid = seq.shape[1] - (step + 1)
            if valid > 0:
                out[:, :valid, step] = seq[:, step + 1 :]
        return out

    def forward(
        self,
        feat,
        history_mask,
        agent_mask,
        scale,
        graph=None,
        future=None,
        mode="train",
        external_anchor=None,
        external_scale=None,
    ):
        if graph is None:
            graph = build(feat, history_mask, rad=DEFAULT_AGENT_RADIUS_METERS, topk=9, lag=1, span=1)

        # normalize_inputs validates shapes and standardizes batch dimensions.
        batch = normalize_inputs(
            feat,
            history_mask,
            agent_mask,
            scale,
            future,
        )
        if external_scale is not None:
            batch.scale = external_scale.to(scale.dtype)
        # Use either the prebuilt batch graph or the legacy on-the-fly graph.
        batch.graph = graph

        # encode_context builds anchors and scene embeddings from history.
        ctx = encode_context(self.encoder, batch.feat, batch.graph)
        if external_anchor is not None:
            # Overwrite calculated anchor with the actual one from dataset (x, y components)
            ctx.anchor = external_anchor[..., :2].to(ctx.anchor.dtype).unsqueeze(1)
            
        # sample_latent_state selects posterior or prior latents for each expert.
        latent = sample_latent_state(self, batch, ctx, mode)
        # decode_predictions converts embeddings and latents into trajectories.
        dec, pred_local, pred_abs = decode_predictions(self, batch, ctx, latent)
        # pack_outputs keeps the legacy output contract unchanged.
        return pack_outputs(batch, ctx, latent, dec, pred_local, pred_abs)

    def safe_init(self):
        for name, p in self.named_parameters():
            if "norm" in name and "weight" in name:
                nn.init.ones_(p)
            elif "norm" in name and "bias" in name:
                nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)
            else:
                if "bias" in name:
                    nn.init.zeros_(p)
                # Keep Mamba/other 1D params as initialized by their respective modules

    def export_onnx(self, path):
        # export_trajectory_model_onnx writes a minimal ONNX wrapper for inference.
        export_trajectory_model_onnx(self, path)
