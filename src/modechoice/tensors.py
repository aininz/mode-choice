from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch

from .config import MODES, ALT2ID, N_ITEMS

@dataclass
class Standardizer:
    feat_names: List[str]
    mu: np.ndarray
    sd: np.ndarray
    feat_names: list

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=np.float32)
        self.sd = np.asarray(self.sd, dtype=np.float32)
        if self.mu.ndim != 1 or self.sd.ndim != 1:
            raise ValueError("mu/sd must be 1D arrays")
        if len(self.mu) != len(self.sd):
            raise ValueError("mu and sd must have same length")    
    
    def transform(self, X_orig: np.ndarray) -> np.ndarray:
        mu = self.mu.reshape(1, 1, -1)
        sd = self.sd.reshape(1, 1, -1)
        return (X_orig - mu) / sd

    @classmethod
    def fit(cls, X_orig: np.ndarray, avail: np.ndarray, feat_names: List[str]) -> "Standardizer":
        mu = np.zeros(len(feat_names), dtype=np.float32)
        sd = np.ones(len(feat_names), dtype=np.float32)
        for k in range(len(feat_names)):
            vals = X_orig[:, :, k][avail == 1]
            mu[k] = float(vals.mean())
            s = float(vals.std())
            sd[k] = s if s > 1e-8 else 1.0
        return cls(feat_names=list(feat_names), mu=mu, sd=sd)    

def build_choice_tensors_hetero(
    df_long: pd.DataFrame,
    item_feat_names: List[str],
    case_feat_names: List[str] = ("income", "urban"),
    scaler: Optional[Standardizer] = None,
    fit_scaler: bool = False,
) -> Tuple[Dict, Standardizer]:
    """
    Builds hetero tensors where per-case vars are repeated per alternative as alt-specific columns:
      base item features: item_feat_names (e.g. cost, ivt, ...)
      + for each mode i: [income_mode_i, urban_mode_i] (or your case_feat_names)
    """
    df_long = df_long.copy()
    cases = np.sort(df_long["case"].unique())
    idx = {c: n for n, c in enumerate(cases)}
    N = len(cases)

    y = np.full(N, -1, dtype=int)
    avail = np.zeros((N, N_ITEMS), dtype=np.float32)

    base_D = len(item_feat_names)

    feat_names = list(item_feat_names)
    for m in MODES:
        for cf in case_feat_names:
            feat_names.append(f"{cf}_{m}")
    D = len(feat_names)

    X_orig = np.zeros((N, N_ITEMS, D), dtype=np.float32)

    for c in cases:
        n = idx[c]
        sub = df_long[df_long["case"] == c]

        # per-case features (take first row)
        case_vals = {}
        for cf in case_feat_names:
            case_vals[cf] = float(sub[cf].iloc[0])

        for _, r in sub.iterrows():
            i = int(r["alt_id"])
            avail[n, i] = 1.0

            X_orig[n, i, :base_D] = r[item_feat_names].to_numpy(np.float32)

            # place hetero vars into that alt's slots
            offset = base_D + i * len(case_feat_names)
            for j, cf in enumerate(case_feat_names):
                X_orig[n, i, offset + j] = case_vals[cf]

            if int(r["choice"]) == 1:
                y[n] = i

    if not (y >= 0).all():
        raise ValueError("Some cases have no chosen alternative.")

    if fit_scaler:
        scaler = Standardizer.fit(X_orig, avail, feat_names)
    else:
        if scaler is None:
            raise ValueError("Provide scaler or set fit_scaler=True on train.")

    X_std = scaler.transform(X_orig)

    tensors = {
        "X_item": torch.tensor(X_std, dtype=torch.float32),
        "X_item_orig": X_orig,  # numpy for perturbation/pricing
        "avail": torch.tensor(avail, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.long),
        "cases": cases,
        "feat_names": scaler.feat_names,
    }
    return tensors, scaler