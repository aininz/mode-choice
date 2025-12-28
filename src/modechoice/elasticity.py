from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from .config import MODES, ALT2ID, ID2ALT, N_ITEMS
from .model import nested_logit_log_probs
from .tensors import Standardizer

@torch.no_grad()
def probs_from_orig(
    X_item_orig: np.ndarray,
    avail_t: torch.Tensor,
    scaler: Standardizer,
    asc: torch.Tensor, beta: torch.Tensor,
    raw_la: torch.Tensor, raw_ll: torch.Tensor,
) -> torch.Tensor:
    Xstd = torch.tensor(scaler.transform(X_item_orig), dtype=torch.float32)
    logP = nested_logit_log_probs(Xstd, avail_t, asc, beta, raw_la, raw_ll)
    P = torch.exp(logP)
    return P / P.sum(dim=1, keepdim=True)

def own_cost_elasticity(
    tensors: dict,
    scaler: Standardizer,
    asc, beta, raw_la, raw_ll,
    mode: str,
    eps=0.01,
) -> float:
    X0 = tensors["X_item_orig"].copy()
    avail = tensors["avail"]
    avail_np = avail.cpu().numpy()
    i = ALT2ID[mode]

    P0 = probs_from_orig(X0, avail, scaler, asc, beta, raw_la, raw_ll)
    D0 = P0.sum(dim=0).cpu().numpy()

    X1 = X0.copy()
    cost_k = scaler.feat_names.index("cost")
    mask = (avail_np[:, i] == 1.0)
    X1[mask, i, cost_k] *= (1.0 + eps)

    P1 = probs_from_orig(X1, avail, scaler, asc, beta, raw_la, raw_ll)
    D1 = P1.sum(dim=0).cpu().numpy()

    e = ((D1[i] - D0[i]) / (D0[i] + 1e-9)) / eps
    return float(e)

def demand_under_multiplier(
    tensors: dict,
    scaler: Standardizer,
    asc, beta, raw_la, raw_ll,
    mode: str,
    mult: float,
) -> np.ndarray:
    X = tensors["X_item_orig"].copy()
    avail = tensors["avail"]
    avail_np = avail.cpu().numpy()

    i = ALT2ID[mode]
    cost_k = scaler.feat_names.index("cost")
    mask = (avail_np[:, i] == 1.0)
    X[mask, i, cost_k] *= float(mult)

    P = probs_from_orig(X, avail, scaler, asc, beta, raw_la, raw_ll)
    return P.sum(dim=0).cpu().numpy()

def response_curve(
    tensors: dict,
    scaler: Standardizer,
    asc, beta, raw_la, raw_ll,
    changed_mode: str,
    mult_grid,
) -> pd.DataFrame:
    D_base = demand_under_multiplier(tensors, scaler, asc, beta, raw_la, raw_ll, changed_mode, 1.0)
    rows = []
    for m in mult_grid:
        D = demand_under_multiplier(tensors, scaler, asc, beta, raw_la, raw_ll, changed_mode, float(m))
        row = {"multiplier": float(m), "mode_changed": changed_mode}
        for j in range(N_ITEMS):
            mode = ID2ALT[j]
            row[f"demand_{mode}"] = float(D[j])
            row[f"rel_demand_{mode}"] = float(D[j] / (D_base[j] + 1e-9))
        rows.append(row)
    return pd.DataFrame(rows)

def plot_substitution_relative(curve_df: pd.DataFrame, changed_mode: str):
    x = curve_df["multiplier"].values
    plt.figure(figsize=(9,4))
    for m in MODES:
        plt.plot(x, curve_df[f"rel_demand_{m}"].values, marker="o", label=m)
    plt.axhline(1.0)
    plt.xlabel(f"Price multiplier for {changed_mode}")
    plt.ylabel("Relative demand (vs baseline)")
    plt.title(f"Substitution response when changing {changed_mode} price")
    plt.legend()
    plt.show()