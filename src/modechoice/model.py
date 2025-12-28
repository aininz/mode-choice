from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import numpy as np

from .config import ALT2ID, MODES, N_ITEMS

def _safe_logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim)

def nested_logit_log_probs(
    X_item: torch.Tensor,          # (N, I, D) standardized
    avail: torch.Tensor,           # (N, I) 0/1
    asc: torch.Tensor,             # (I,)
    beta: torch.Tensor,            # (D,)
    raw_lam_air: torch.Tensor,     # scalar
    raw_lam_land: torch.Tensor,    # scalar
) -> torch.Tensor:
    N, I, D = X_item.shape
    assert I == 4, "Assumes 4 alternatives: train, car, bus, air."

    lam_air  = 0.05 + 0.95 * torch.sigmoid(raw_lam_air)
    lam_land = 0.05 + 0.95 * torch.sigmoid(raw_lam_land)

    V = asc.view(1, I) + torch.einsum("nid,d->ni", X_item, beta)

    neg_inf = torch.tensor(-1e9, device=V.device, dtype=V.dtype)
    V = torch.where(avail > 0, V, neg_inf)

    air_idx  = torch.tensor([ALT2ID["air"]], device=V.device)
    land_idx = torch.tensor([ALT2ID["train"], ALT2ID["car"], ALT2ID["bus"]], device=V.device)

    V_air  = V.index_select(1, air_idx) / lam_air
    V_land = V.index_select(1, land_idx) / lam_land

    lse_air  = _safe_logsumexp(V_air, dim=1)
    lse_land = _safe_logsumexp(V_land, dim=1)

    IV_air  = lam_air  * lse_air
    IV_land = lam_land * lse_land

    IV = torch.stack([IV_land, IV_air], dim=1)  # land, air
    logP_nest = IV - _safe_logsumexp(IV, dim=1).unsqueeze(1)

    logP_air_given  = V_air  - lse_air.unsqueeze(1)
    logP_land_given = V_land - lse_land.unsqueeze(1)

    logP = torch.full((N, I), neg_inf, device=V.device, dtype=V.dtype)

    logP_land = logP_nest[:, 0].unsqueeze(1) + logP_land_given
    logP.scatter_(1, land_idx.view(1, -1).expand(N, -1), logP_land)

    logP_air = logP_nest[:, 1].unsqueeze(1) + logP_air_given
    logP.scatter_(1, air_idx.view(1, -1).expand(N, -1), logP_air)

    logP = torch.where(avail > 0, logP, neg_inf)
    return logP

@dataclass
class NestedLogitHetero:
    asc: torch.Tensor
    beta: torch.Tensor
    raw_lam_air: torch.Tensor
    raw_lam_land: torch.Tensor

    def lambdas(self) -> Dict[str, float]:
        la = float((0.05 + 0.95 * torch.sigmoid(self.raw_lam_air)).detach().cpu().item())
        ll = float((0.05 + 0.95 * torch.sigmoid(self.raw_lam_land)).detach().cpu().item())
        return {"air": la, "land": ll}

    @torch.no_grad()
    def predict_proba(self, X_item: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        logP = nested_logit_log_probs(X_item, avail, self.asc, self.beta, self.raw_lam_air, self.raw_lam_land)
        P = torch.exp(logP)
        return P / P.sum(dim=1, keepdim=True)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "asc": self.asc.detach().cpu(),
            "beta": self.beta.detach().cpu(),
            "raw_lam_air": self.raw_lam_air.detach().cpu(),
            "raw_lam_land": self.raw_lam_land.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, sd: Dict[str, torch.Tensor]) -> "NestedLogitHetero":
        return cls(
            asc=sd["asc"].clone(),
            beta=sd["beta"].clone(),
            raw_lam_air=sd["raw_lam_air"].clone(),
            raw_lam_land=sd["raw_lam_land"].clone(),
        )