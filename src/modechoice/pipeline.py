from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch

from .config import MODES
from .model import NestedLogitHetero, nested_logit_log_probs
from .train import fit_nested_logit
from .eval import eval_basic, eval_classification
from .tensors import Standardizer


@dataclass
class ModeChoicePipeline:
    """
    Pipeline for model:
      - holds model + scaler + metadata
      - can evaluate and produce coefficient tables
    """
    model: NestedLogitHetero
    scaler: Standardizer
    meta: Dict

    @classmethod
    def fit(
        cls,
        train_t: dict,
        val_t: dict,
        scaler: Standardizer,
        meta: Optional[Dict] = None,
        lr: float = 0.05,
        max_epochs: int = 800,
        patience: int = 25,
        seed: int = 0,
    ) -> "ModeChoicePipeline":
        asc_f, beta_f, raw_la_f, raw_ll_f = fit_nested_logit(
            train_t, val_t, lr=lr, max_epochs=max_epochs, patience=patience, seed=seed
        )
        model = NestedLogitHetero(asc_f, beta_f, raw_la_f, raw_ll_f)
        return cls(model=model, scaler=scaler, meta=(meta or {}))

    # -------------------------
    # Predict / evaluate
    # -------------------------
    def predict_proba(self, tensors: dict) -> torch.Tensor:
        return self.model.predict_proba(tensors["X_item"], tensors["avail"])

    def evaluate(self, name: str, tensors: dict, classification: bool = True) -> None:
        asc, beta = self.model.asc, self.model.beta
        raw_la, raw_ll = self.model.raw_lam_air, self.model.raw_lam_land
        eval_basic(name, tensors, asc, beta, raw_la, raw_ll)
        if classification:
            eval_classification(name, tensors, asc, beta, raw_la, raw_ll)

    def evaluate_all(self, split_dict: Dict[str, dict], classification: bool = True) -> None:
        """
        split_dict = {"TRAIN": train_t, "VAL": val_t, "TEST": test_t}
        """
        for name, t in split_dict.items():
            self.evaluate(name, t, classification=classification)

    @torch.no_grad()
    def _basic_metrics_row(self, name: str, tensors: dict) -> Dict:
        """
        Returns a single-row dict with:
          - NLL
          - accuracy
          - pred_share_* and actual_share_*
        """
        asc, beta = self.model.asc, self.model.beta
        raw_la, raw_ll = self.model.raw_lam_air, self.model.raw_lam_land

        logP = nested_logit_log_probs(tensors["X_item"], tensors["avail"], asc, beta, raw_la, raw_ll)
        P = torch.exp(logP)
        P = P / P.sum(dim=1, keepdim=True)

        y = tensors["y"]
        nll = -(logP[torch.arange(len(y)), y]).mean().item()
        acc = (P.argmax(dim=1) == y).float().mean().item()

        pred_share = P.mean(dim=0).cpu().numpy()
        actual = pd.Series(y.cpu().numpy()).value_counts(normalize=True).sort_index()
        actual_share = np.array([actual.get(i, 0.0) for i in range(len(MODES))])

        row = {"split": name, "nll": float(nll), "accuracy": float(acc)}
        for i, m in enumerate(MODES):
            row[f"pred_share_{m}"] = float(pred_share[i])
            row[f"actual_share_{m}"] = float(actual_share[i])
        return row

    def evaluate_summary_all(
        self,
        split_dict: Dict[str, dict],
        detailed_on: str = "TEST",
        show_detailed: bool = True,
    ) -> pd.DataFrame:
        """
        - For ALL splits: returns a dataframe with NLL/accuracy/pred vs actual shares.
        - For ONE split (default TEST): prints confusion matrix + classification metrics.
        """
        rows: List[Dict] = [self._basic_metrics_row(name, t) for name, t in split_dict.items()]
        summary_df = pd.DataFrame(rows)

        # keep split ordering nice if you pass TRAIN/VAL/TEST
        order = ["TRAIN", "VAL", "TEST"]
        if "split" in summary_df.columns:
            summary_df["split"] = pd.Categorical(summary_df["split"], categories=order, ordered=True)
            summary_df = summary_df.sort_values("split")

        if show_detailed and (detailed_on in split_dict):
            self.evaluate(detailed_on, split_dict[detailed_on], classification=True)

        return summary_df.reset_index(drop=True)

    # -------------------------
    # Coefficients tables
    # -------------------------
    def coef_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns (asc_df, beta_df, lambda_df)
        """
        asc = self.model.asc.detach().cpu().numpy()
        beta = self.model.beta.detach().cpu().numpy()

        feat_names = self.meta.get("feat_names") or self.scaler.feat_names

        asc_df = pd.DataFrame({"mode": MODES, "ASC": asc}).sort_values("ASC", ascending=False)

        beta_df = pd.DataFrame({"feature": feat_names, "beta": beta})
        beta_df["abs_beta"] = np.abs(beta_df["beta"])
        beta_df = beta_df.sort_values("abs_beta", ascending=False).drop(columns=["abs_beta"])

        lam = self.model.lambdas()
        lam_df = pd.DataFrame({"nest": ["air", "land"], "lambda": [lam["air"], lam["land"]]})

        return asc_df, beta_df, lam_df

    def coef_tables_original_units(self) -> pd.DataFrame:
        """
        Converts standardized betas to "per 1 unit" in original scale:
          beta_orig = beta_std / sd
        """
        beta_std = self.model.beta.detach().cpu().numpy()
        sd = np.asarray(self.scaler.sd, dtype=float)
        feat_names = self.scaler.feat_names

        beta_orig = beta_std / sd

        df = pd.DataFrame(
            {
                "feature": feat_names,
                "beta_std": beta_std,
                "sd": sd,
                "beta_per_1_unit": beta_orig,
            }
        )
        df["abs_beta_per_1_unit"] = np.abs(df["beta_per_1_unit"])
        df = df.sort_values("abs_beta_per_1_unit", ascending=False).drop(columns=["abs_beta_per_1_unit"])
        return df