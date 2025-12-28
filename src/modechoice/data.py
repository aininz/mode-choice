from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import MODES, ALT2ID

def ensure_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures:
      - alt lowercased and in MODES
      - alt_id exists
      - numeric columns coerced where relevant
      - one chosen alt per case (filters bad cases)
    """
    out = df.copy()
    out["alt"] = out["alt"].astype(str).str.lower().str.strip()
    out = out[out["alt"].isin(MODES)].copy()
    out["alt_id"] = out["alt"].map(ALT2ID).astype(int)

    # coerce basics (extend as needed)
    for c in ["case", "choice", "cost", "ivt", "ovt", "freq", "income", "urban"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["choice"] = (out["choice"] > 0).astype(int)
    out = out.dropna(subset=["case", "alt_id", "choice"]).copy()

    # remove duplicates per (case, alt)
    out = out.sort_values(["case", "alt_id"]).drop_duplicates(["case", "alt_id"], keep="first")

    # keep only cases with exactly one chosen
    chosen = out.groupby("case")["choice"].sum()
    good_cases = chosen[chosen == 1].index
    out = out[out["case"].isin(good_cases)].copy()
    return out

def stratified_split_by_case(df_long: pd.DataFrame, seed=42, test_size=0.3, val_share_of_temp=0.5):
    chosen = (df_long[df_long["choice"] == 1]
              .drop_duplicates("case")[["case","alt_id"]])

    cases = chosen["case"].to_numpy()
    ycase = chosen["alt_id"].to_numpy()

    train_cases, temp_cases = train_test_split(
        cases, test_size=test_size, random_state=seed, shuffle=True, stratify=ycase
    )

    temp_y = chosen.set_index("case").loc[temp_cases, "alt_id"].to_numpy()
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=1 - val_share_of_temp, random_state=seed, shuffle=True, stratify=temp_y
    )
    return train_cases, val_cases, test_cases
