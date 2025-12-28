from __future__ import annotations

import sys
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]   # repo root (mode-choice/)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modechoice.config import MODES, ALT2ID, ID2ALT, N_ITEMS
from modechoice.data import ensure_long_format
from modechoice.features import add_relative_features_long
from modechoice.tensors import build_choice_tensors_hetero, Standardizer
from modechoice.io import load_bundle
from modechoice.pipeline import ModeChoicePipeline
from modechoice.model import NestedLogitHetero, nested_logit_log_probs

# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Substitution Response", layout="wide")
st.title("📈 Substitution Response")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def infer_item_and_case_feat_names(feat_names: List[str]) -> Tuple[List[str], List[str]]:
    """
    Given scaler.feat_names like:
      ['cost','ivt','ovt','freq','gen_time','rel_cost', ..., 'income_train','urban_train', ...]
    infer:
      item_feat_names = everything that is NOT '<casefeat>_<mode>'
      case_feat_names = unique prefixes (case feats), e.g. ['income','urban']
    """
    case_feat_names: List[str] = []
    for fn in feat_names:
        for m in MODES:
            suf = f"_{m}"
            if fn.endswith(suf):
                pref = fn[: -len(suf)]
                if pref not in case_feat_names:
                    case_feat_names.append(pref)

    case_cols = {f"{cf}_{m}" for cf in case_feat_names for m in MODES}
    item_feat_names = [fn for fn in feat_names if fn not in case_cols]
    return item_feat_names, case_feat_names


@st.cache_resource
def load_pipe(bundle_path: str) -> ModeChoicePipeline:
    """
    Cache the loaded pipeline (model+scaler+meta). This is a resource, not data.
    """
    pipe = load_bundle(
        bundle_path,
        PipelineCls=ModeChoicePipeline,
        ModelCls=NestedLogitHetero,
        ScalerCls=Standardizer,
    )
    return pipe


@st.cache_data
def load_all_t(
    data_path: str,
    w_ovt: float,
    freq_period_minutes: float,
    bundle_path: str,
) -> dict:
    """
    Cache tensors built from the dataset. We key by (data_path, feature params, bundle_path).
    IMPORTANT: we do NOT pass `pipe` directly (Streamlit can't hash it).
    """
    pipe = load_pipe(bundle_path)

    df_raw = pd.read_csv(data_path)
    df_long = ensure_long_format(df_raw)

    df_long = add_relative_features_long(
        df_long,
        w_ovt=float(w_ovt),
        freq_period_minutes=float(freq_period_minutes),
    )

    item_feat_names, case_feat_names = infer_item_and_case_feat_names(pipe.scaler.feat_names)

    all_t, _ = build_choice_tensors_hetero(
        df_long,
        item_feat_names=item_feat_names,
        case_feat_names=case_feat_names,
        scaler=pipe.scaler,
        fit_scaler=False,
    )

    # sanity
    if not isinstance(all_t, dict):
        raise TypeError(f"Expected dict tensors, got {type(all_t)}")
    if "X_item_orig" not in all_t:
        raise KeyError(f"Expected key 'X_item_orig'. Keys: {list(all_t.keys())}")

    return all_t


@torch.no_grad()
def probs_from_orig(X_item_orig: np.ndarray, avail_t: torch.Tensor, pipe: ModeChoicePipeline) -> torch.Tensor:
    """
    Given original-scale X (numpy), compute P for each case.
    """
    Xstd = torch.tensor(pipe.scaler.transform(X_item_orig), dtype=torch.float32)
    logP = nested_logit_log_probs(
        Xstd,
        avail_t,
        pipe.model.asc,
        pipe.model.beta,
        pipe.model.raw_lam_air,
        pipe.model.raw_lam_land,
    )
    P = torch.exp(logP)
    return P / P.sum(dim=1, keepdim=True)


def demand_under_multiplier(tensors: dict, pipe: ModeChoicePipeline, mode: str, mult: float) -> np.ndarray:
    """
    Returns expected demand vector across modes = sum_n P(n, i).
    """
    X = tensors["X_item_orig"].copy()
    avail = tensors["avail"]
    avail_np = avail.cpu().numpy()

    i = ALT2ID[mode]
    cost_k = pipe.scaler.feat_names.index("cost")

    mask = (avail_np[:, i] == 1.0)
    X[mask, i, cost_k] *= float(mult)

    P = probs_from_orig(X, avail, pipe)
    return P.sum(dim=0).cpu().numpy()


def response_curve(tensors: dict, pipe: ModeChoicePipeline, changed_mode: str, mult_grid: np.ndarray) -> pd.DataFrame:
    """
    Build response curve dataframe for changing ONE mode's price.
    """
    D_base = demand_under_multiplier(tensors, pipe, changed_mode, 1.0)

    rows = []
    for mm in mult_grid:
        D = demand_under_multiplier(tensors, pipe, changed_mode, float(mm))
        row = {"multiplier": float(mm), "mode_changed": changed_mode}
        for j in range(N_ITEMS):
            mode = ID2ALT[j]
            row[f"demand_{mode}"] = float(D[j])
            row[f"rel_demand_{mode}"] = float(D[j] / (D_base[j] + 1e-9))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_relative(curve_df: pd.DataFrame, changed_mode: str, display_modes: List[str]):
    x = curve_df["multiplier"].values
    fig = plt.figure(figsize=(9, 4))
    for m in display_modes:
        plt.plot(x, curve_df[f"rel_demand_{m}"].values, marker="o", label=m)
    plt.axhline(1.0)
    plt.xlabel(f"Price multiplier for {changed_mode}")
    plt.ylabel("Relative demand (vs baseline)")
    plt.title(f"Substitution response when changing {changed_mode} price")
    plt.legend()
    return fig


def plot_absolute(curve_df: pd.DataFrame, changed_mode: str, display_modes: List[str]):
    x = curve_df["multiplier"].values
    fig = plt.figure(figsize=(9, 4))
    for m in display_modes:
        plt.plot(x, curve_df[f"demand_{m}"].values, marker="o", label=m)
    plt.xlabel(f"Price multiplier for {changed_mode}")
    plt.ylabel("Expected demand (sum of probabilities)")
    plt.title(f"Absolute expected demand when changing {changed_mode} price")
    plt.legend()
    return fig



# -----------------------------------------------------------------------------
# Sidebar UI
# -----------------------------------------------------------------------------
with st.sidebar:
    # st.header("Inputs")

    default_bundle = ROOT / "artifacts" / "modechoice_bundle"
    # BUNDLE_PATH = st.text_input("Bundle path", value=str(default_bundle))
    BUNDLE_PATH = str(default_bundle)

    default_data = ROOT / "dataset" / "ModeCanada.csv"
    # DATA_PATH = st.text_input("Data CSV path", value=str(default_data))
    DATA_PATH = str(default_data)

    st.subheader("Feature engineering")
    w_ovt = st.number_input("w_ovt (weight for ovt in gen_time)", value=2.0, step=0.5)
    freq_period_minutes = st.number_input("freq_period_minutes", value=1440.0, step=60.0)

    st.divider()
    st.subheader("Curve parameters")

    changed_mode = st.selectbox(
        "Mode to change price",
        options=MODES,
        index=MODES.index("air") if "air" in MODES else 0,
    )

    display_modes = st.multiselect(
        "Modes to display",
        options=MODES,
        default=MODES,
    )
    if not display_modes:
        st.warning("Pick at least one mode to display.")
        display_modes = [changed_mode]

    lo = st.number_input("Multiplier min", value=0.7, step=0.05)
    hi = st.number_input("Multiplier max", value=1.3, step=0.05)
    n_points = st.number_input("Number of points", value=25, min_value=5, max_value=101, step=1)

    show_abs = st.checkbox("Also show absolute expected demand", value=False)

# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
try:
    pipe = load_pipe(BUNDLE_PATH)
except Exception as e:
    st.error(f"Failed to load bundle: {e}")
    st.stop()

st.caption(f"Loaded model lambdas: {pipe.model.lambdas()}")

try:
    all_t = load_all_t(DATA_PATH, w_ovt, freq_period_minutes, BUNDLE_PATH)
except Exception as e:
    st.error(f"Failed to build tensors from dataset: {e}")
    st.stop()

mult_grid = np.linspace(float(lo), float(hi), int(n_points))
curve_df = response_curve(all_t, pipe, changed_mode, mult_grid)

# table columns only for chosen display modes
cols = ["multiplier", "mode_changed"]
for m in display_modes:
    cols += [f"demand_{m}", f"rel_demand_{m}"]
table_df = curve_df[cols].round(4)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Relative demand curves")
    fig = plot_relative(curve_df, changed_mode, display_modes)
    st.pyplot(fig, clear_figure=True)

    if show_abs:
        st.subheader("Absolute expected demand")
        fig2 = plot_absolute(curve_df, changed_mode, display_modes)
        st.pyplot(fig2, clear_figure=True)

with col2:
    st.subheader("Table")
    st.dataframe(table_df, use_container_width=True)