from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional
import numpy as np
import torch

from .config import MODES, ALT2ID, N_ITEMS
from .model import nested_logit_log_probs
from .tensors import Standardizer

def make_user_tensor_hetero(
    base_item: dict,                  # mode -> item feature dict
    item_feat_names: list[str],       # item-level feats used in training (no income_*/urban_*)
    case_features: dict,              # {"income": 70, "urban": 1}
    scaler: Standardizer,
):
    """
    Builds a single-user (N=1) tensor in EXACT feature order scaler.feat_names.
    Works even if item features include extra engineered terms like gen_time, urban_x_ovt, etc.
    """
    feat_names = list(scaler.feat_names)
    D = len(feat_names)
    I = len(MODES)

    idx = {n:i for i, n in enumerate(feat_names)}

    X_orig = np.zeros((1, I, D), dtype=np.float32)
    avail  = np.zeros((1, I), dtype=np.float32)

    # fill per mode
    for i, m in enumerate(MODES):
        if m not in base_item:
            continue

        avail[0, i] = 1.0

        # item feats
        for f in item_feat_names:
            if f not in idx:
                continue
            if f not in base_item[m]:
                X_orig[0, i, idx[f]] = 0.0
            else:
                X_orig[0, i, idx[f]] = float(base_item[m][f])

        for cf in ["income", "urban"]:
            name = f"{cf}_{m}"
            if name in idx:
                X_orig[0, i, idx[name]] = float(case_features.get(cf, 0.0))

    X_std = scaler.transform(X_orig)

    return (
        torch.tensor(X_std, dtype=torch.float32),
        torch.tensor(avail, dtype=torch.float32),
        torch.tensor(X_orig, dtype=torch.float32),
    )

@torch.no_grad()
def user_probs(Xstd, avail, asc, beta, raw_la, raw_ll) -> np.ndarray:
    logP = nested_logit_log_probs(Xstd, avail, asc, beta, raw_la, raw_ll)
    P = torch.exp(logP); P = P / P.sum(dim=1, keepdim=True)
    return P[0].cpu().numpy()

def optimize_user(
    X_user_orig_t: torch.Tensor,
    avail_user_t: torch.Tensor,
    scaler: Standardizer,
    asc, beta, raw_la, raw_ll,
    control_modes: Sequence[str],
    target_modes: Sequence[str],
    objective: str = "revenue",   # "revenue" or "prob"
    mult_bounds=(0.8, 1.2),
    steps=250,
    lr=0.08,
    forbid_modes: Optional[Sequence[str]] = ("car",),  # travel agency doesn't sell car tickets
):
    """
    control_modes: which modes YOU can price
    target_modes: which modes you want to push (max 3 recommended)
    objective:
      - "prob": maximize sum P(target_modes)
      - "revenue": maximize sum (price_i * P_i) over the modes you SELL (typically control_modes or target_modes)
    forbid_modes: modes you’re not allowed to “optimize toward” (e.g. car)
    """
    control_modes = list(control_modes)
    target_modes = list(target_modes)

    if forbid_modes:
        for fm in forbid_modes:
            if fm in target_modes:
                raise ValueError(f"target_modes contains forbidden mode '{fm}'. Remove it.")

    if len(target_modes) == 0 or len(target_modes) > 3:
        raise ValueError("target_modes must contain 1 to 3 modes.")

    feat_names = scaler.feat_names
    D = len(feat_names)
    cost_k = feat_names.index("cost")

    control_ids = [ALT2ID[m] for m in control_modes]
    target_ids = torch.tensor([ALT2ID[m] for m in target_modes], dtype=torch.long)

    lo, hi = mult_bounds
    theta = torch.zeros(len(control_ids), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)

    mu = torch.tensor(scaler.mu, dtype=torch.float32).view(1, 1, D)
    sd = torch.tensor(scaler.sd, dtype=torch.float32).view(1, 1, D)

    # baseline prices for revenue objective
    base_cost = X_user_orig_t[0, :, cost_k].detach().cpu().numpy()

    for _ in range(steps):
        opt.zero_grad()

        m_ctrl = lo + (hi - lo) * torch.sigmoid(theta)  # bounded
        mult_all = torch.ones(N_ITEMS)

        for j, i in enumerate(control_ids):
            mult_all[i] = m_ctrl[j]

        mult_mat = 1.0 + (mult_all.view(1, N_ITEMS) - 1.0) * avail_user_t
        X_cost_scaled = X_user_orig_t[:, :, cost_k] * mult_mat

        cols = []
        for kk in range(D):
            cols.append((X_cost_scaled if kk == cost_k else X_user_orig_t[:, :, kk]).unsqueeze(2))
        X1 = torch.cat(cols, dim=2)

        Xstd = (X1 - mu) / sd
        logP = nested_logit_log_probs(Xstd, avail_user_t, asc, beta, raw_la, raw_ll)
        P = torch.exp(logP); P = P / P.sum(dim=1, keepdim=True)

        if objective == "prob":
            obj = P[0, target_ids].sum()
        elif objective == "revenue":
            # simple expected revenue: sum price_i * P_i (only for controlled/targeted modes)
            sell_ids = torch.tensor(sorted(set(control_ids)), dtype=torch.long)
            prices = X1[0, :, cost_k]
            obj = (prices[sell_ids] * P[0, sell_ids]).sum()
        else:
            raise ValueError("objective must be 'prob' or 'revenue'.")

        (-obj).backward()
        opt.step()

    with torch.no_grad():
        m_ctrl = lo + (hi - lo) * torch.sigmoid(theta)
        mult_all = torch.ones(N_ITEMS)
        for j, i in enumerate(control_ids):
            mult_all[i] = m_ctrl[j]

        mult_mat = 1.0 + (mult_all.view(1, N_ITEMS) - 1.0) * avail_user_t
        X_cost_scaled = X_user_orig_t[:, :, cost_k] * mult_mat

        cols = []
        for kk in range(D):
            cols.append((X_cost_scaled if kk == cost_k else X_user_orig_t[:, :, kk]).unsqueeze(2))
        X1 = torch.cat(cols, dim=2)
        Xstd = (X1 - mu) / sd

        logP = nested_logit_log_probs(Xstd, avail_user_t, asc, beta, raw_la, raw_ll)
        P = torch.exp(logP); P = P / P.sum(dim=1, keepdim=True)
        probs = P[0].cpu().numpy()

        new_cost = X1[0, :, cost_k].cpu().numpy()

    return mult_all.cpu().numpy(), base_cost, new_cost, probs