import torch
from torch import nn


def export_trajectory_model_onnx(model, path):
    class Wrap(nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.model = wrapped_model

        def forward(self, feat, sa, st, da, dt, scale):
            steps = feat.shape[2]
            src_node = sa.to(torch.long) * steps + st.to(torch.long)
            dst_node = da.to(torch.long) * steps + dt.to(torch.long)
            edge = {"src_node": src_node, "dst_node": dst_node}
            history_mask = torch.ones(feat.shape[:3], dtype=torch.bool, device=feat.device)
            agent_mask = torch.ones(feat.shape[:2], dtype=torch.bool, device=feat.device)
            out = self.model(
                feat,
                history_mask,
                agent_mask,
                scale,
                graph={
                    "history_mask": history_mask,
                    "agent_mask": agent_mask,
                    "temporal": edge,
                    "spatial": edge,
                    "cross": edge,
                },
                future=None,
                mode="eval",
            )
            return out["y_all"].mean(dim=1)

    agent_count = 2
    history_steps = 50
    feat_dim = model.encoder.inp.net[0].in_features
    feat = torch.zeros((1, agent_count, history_steps, feat_dim), dtype=torch.float16)
    sa = torch.tensor([0], dtype=torch.long)
    st = torch.tensor([0], dtype=torch.long)
    da = torch.tensor([0], dtype=torch.long)
    dt = torch.tensor([0], dtype=torch.long)
    scale = torch.ones((1, 1, 1), dtype=torch.float16)
    torch.onnx.export(
        Wrap(model),
        (feat, sa, st, da, dt, scale),
        path,
        input_names=[
            "feat",
            "src_agent",
            "src_time",
            "dst_agent",
            "dst_time",
            "scale",
        ],
        output_names=["pred"],
        dynamic_axes={
            "feat": {0: "batch"},
            "scale": {0: "batch"},
        },
        opset_version=17,
    )
