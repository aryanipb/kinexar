from dataclasses import dataclass
from typing import Dict, Optional

from torch import Tensor


Graph = Dict[str, Dict[str, Tensor]]


@dataclass
class NormalizedBatch:
    feat: Tensor
    graph: Graph
    scale: Tensor
    future: Optional[Tensor]
    squeeze: bool


@dataclass
class ModelContext:
    history_mask: Tensor
    anchor: Tensor
    h_nodes: Tensor
    h: Tensor
    ctx_mask: Tensor
    h_global: Tensor
    ctx_mean: Tensor


@dataclass
class LatentState:
    future_local: Optional[Tensor]
    prior_mu: Optional[Tensor]
    prior_logvar: Optional[Tensor]
    post_mu: Optional[Tensor]
    post_logvar: Optional[Tensor]
    z_expert: Tensor
