from __future__ import annotations
from typing import Iterable, List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import torch

from .config import MODES, ALT2ID, N_ITEMS
from .model import nested_logit_log_probs
from .tensors import Standardizer

def make_mask(control_modes: Iterable[str], modes: List[str] = MODES) -> torch.Tensor:
    mask = torch.zeros(len(modes), dtype=torch.bool)
    for m in control_modes:
        mask[ALT2ID[m]] = True
    return mask

def revenue_with_cap_diff_scenario(
    tensors: dict,
    scaler: Standardizer,
    asc, beta, raw_la, raw_ll,
    theta: torch.Tensor,          # (I,) unconstrained
    cap: torch.Tensor,            # (I,)
    scenario_mult: torch.Tensor,  # (I,)
    controllable_mask: Optional[torch.Tensor] = None,
    k_smooth=30.0,
):
    X0 = torch.as_tensor(tensors["X_item_orig"], dtype=torch.float32)  # (N,I,D)
    avail = tensors["avail"]                                           # (N,I)
    N, I, D = X0.shape
    cost_k = scaler.feat_names.index("cost")

    decision_mult = torch.exp(theta)  # positive

    if controllable_mask is None:
        total_mult = scenario_mult
        decision_mult_vis = decision_mult
    else:
        # controlled -> decision, uncontrolled -> scenario
        total_mult = torch.where(controllable_mask, decision_mult, scenario_mult)
        decision_mult_vis = torch.where(controllable_mask, decision_mult, torch.ones_like(decision_mult))

    # broadcast multipliers only where available
    mult_mat = 1.0 + (total_mult.view(1, I) - 1.0) * avail

    X_cost_scaled = X0[:, :, cost_k] * mult_mat

    cols = []
    for j in range(D):
        cols.append((X_cost_scaled if j == cost_k else X0[:, :, j]).unsqueeze(2))
    X1 = torch.cat(cols, dim=2)

    mu = torch.tensor(scaler.mu, dtype=torch.float32).view(1, 1, D)
    sd = torch.tensor(scaler.sd, dtype=torch.float32).view(1, 1, D)
    Xstd = (X1 - mu) / sd

    logP = nested_logit_log_probs(Xstd, avail, asc, beta, raw_la, raw_ll)
    P = torch.exp(logP); P = P / P.sum(dim=1, keepdim=True)

    Dexp = P.sum(dim=0)

    price_i = (X1[:, :, cost_k] * avail).sum(dim=0) / (avail.sum(dim=0).clamp_min(1.0))

    sold = Dexp - torch.nn.functional.softplus(Dexp - cap) / k_smooth

    rev = (price_i * sold).sum()
    return rev, Dexp, sold, price_i, total_mult, decision_mult_vis

def run_scenario_grid(
    tensors: dict,
    scaler: Standardizer,
    asc, beta, raw_la, raw_ll,
    cap: torch.Tensor,
    scenario_mult_list: List[List[float]],
    controllable_mask: Optional[torch.Tensor],
    steps=250,
    lr=0.05,
    k_smooth=30.0,
    warm_start=True,
    verbose_every=0,
) -> pd.DataFrame:
    results = []
    I = N_ITEMS
    theta_prev = torch.zeros(I)

    for s_idx, scen in enumerate(scenario_mult_list):
        scenario_mult = torch.tensor(scen, dtype=torch.float32)
        assert scenario_mult.shape[0] == I, "scenario_mult must be length 4: [train, car, bus, air]"

        theta = (theta_prev.clone().detach().requires_grad_(True) if warm_start
                 else torch.zeros(I, requires_grad=True))

        opt = torch.optim.Adam([theta], lr=lr)

        for step in range(steps):
            opt.zero_grad()
            rev, Dexp, sold, price_i, total_mult, decision_mult = revenue_with_cap_diff_scenario(
                tensors=tensors,
                scaler=scaler,
                asc=asc, beta=beta, raw_la=raw_la, raw_ll=raw_ll,
                theta=theta,
                cap=cap,
                scenario_mult=scenario_mult,
                controllable_mask=controllable_mask,
                k_smooth=k_smooth,
            )
            (-rev).backward()
            opt.step()

            if verbose_every and (step % verbose_every == 0):
                print(f"[scenario {s_idx}] step {step:3d} rev={rev.item():.2f} "
                      f"dec={decision_mult.detach().cpu().numpy().round(3)}")

        with torch.no_grad():
            rev, Dexp, sold, price_i, total_mult, decision_mult = revenue_with_cap_diff_scenario(
                tensors=tensors,
                scaler=scaler,
                asc=asc, beta=beta, raw_la=raw_la, raw_ll=raw_ll,
                theta=theta,
                cap=cap,
                scenario_mult=scenario_mult,
                controllable_mask=controllable_mask,
                k_smooth=k_smooth,
            )

        theta_prev = theta.detach()

        row = {"scenario_id": s_idx, "revenue_total": float(rev.item())}
        for i, m in enumerate(MODES):
            row[f"scenario_mult_{m}"] = float(scenario_mult[i].item())
            row[f"decision_mult_{m}"] = float(decision_mult[i].item())
            row[f"total_mult_{m}"] = float(total_mult[i].item())
            row[f"avg_price_{m}"] = float(price_i[i].item())
            row[f"demand_{m}"] = float(Dexp[i].item())
            row[f"sold_{m}"] = float(sold[i].item())
            row[f"revenue_{m}"] = float((price_i[i] * sold[i]).item())
        results.append(row)

    return pd.DataFrame(results)