# app/pages/2_Population_Pricing.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modechoice.config import MODES, N_ITEMS
from modechoice.data import ensure_long_format
from modechoice.features import add_relative_features_long
from modechoice.tensors import build_choice_tensors_hetero, Standardizer
from modechoice.io import load_bundle
from modechoice.pipeline import ModeChoicePipeline
from modechoice.model import NestedLogitHetero
from modechoice.pricing_population import make_mask, run_scenario_grid

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Population Pricing", layout="wide")
st.title("👥 Population Pricing")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def infer_item_and_case_feat_names(feat_names: List[str]) -> Tuple[List[str], List[str]]:
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


def parse_scenarios(text: str, modes: List[str]) -> List[List[float]]:
    """
    Accept rows like:
      1.00,1.00,1.00,1.00
      1.10,1.05,1.00,0.95
    Order must match MODES.
    """
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) != len(modes):
            raise ValueError(f"Each scenario row must have {len(modes)} values (order={modes}). Got {len(parts)}: {line}")
        rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError("No scenarios parsed.")
    return rows


def scenarios_to_df(scenarios: List[List[float]], modes: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(scenarios, columns=[f"scenario_mult_{m}" for m in modes])
    df.insert(0, "scenario_id", np.arange(len(df), dtype=int))
    return df


def pretty_cap_table(cap: torch.Tensor) -> pd.DataFrame:
    return pd.DataFrame({"mode": MODES, "capacity": [float(x) for x in cap.cpu().numpy()]})


@st.cache_resource
def load_pipe(bundle_path: str) -> ModeChoicePipeline:
    return load_bundle(
        bundle_path,
        PipelineCls=ModeChoicePipeline,
        ModelCls=NestedLogitHetero,
        ScalerCls=Standardizer,
    )


@st.cache_data
def load_all_t(
    data_path: str,
    w_ovt: float,
    freq_period_minutes: float,
    bundle_path: str,
) -> dict:
    """
    Same caching strategy as Substitution Response:
      - do NOT pass pipe directly (Streamlit can't hash it)
      - use bundle_path as cache key, load pipe inside
    """
    pipe = load_pipe(bundle_path)

    df_raw = pd.read_csv(data_path)
    df_long = ensure_long_format(df_raw)
    df_long = add_relative_features_long(df_long, w_ovt=float(w_ovt), freq_period_minutes=float(freq_period_minutes))

    item_feat_names, case_feat_names = infer_item_and_case_feat_names(pipe.scaler.feat_names)

    all_t, _ = build_choice_tensors_hetero(
        df_long,
        item_feat_names=item_feat_names,
        case_feat_names=case_feat_names,
        scaler=pipe.scaler,
        fit_scaler=False,
    )
    return all_t


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    # st.header("Inputs")

    default_bundle = ROOT / "artifacts" / "modechoice_bundle"
    # BUNDLE_PATH = st.text_input("Bundle path", value=str(default_bundle))
    BUNDLE_PATH = default_bundle

    default_data = ROOT / "dataset" / "ModeCanada.csv"
    # DATA_PATH = st.text_input("Data CSV path", value=str(default_data))
    DATA_PATH = default_data

    st.subheader("Controllable modes")
    control_modes = st.multiselect(
        "You can price these modes",
        options=MODES,
        default=["train", "air"] if ("train" in MODES and "air" in MODES) else [MODES[0]],
    )

    st.divider()
    st.subheader("Feature engineering")
    w_ovt = st.number_input("w_ovt", value=2.0, step=0.5)
    freq_period_minutes = st.number_input("freq_period_minutes", value=1440.0, step=60.0)

    st.divider()
    st.subheader("Capacity")
    st.caption("Capacity of each mode (used by the smooth cap constraint).")

    # default-ish capacities (can be tweaked)
    cap_defaults = {m: 1000.0 for m in MODES}
    cap_inputs = {}
    for m in MODES:
        cap_inputs[m] = st.number_input(f"cap_{m}", value=float(cap_defaults[m]), step=50.0)

    st.divider()
    st.subheader("Optimization")
    steps = st.number_input("steps", value=250, min_value=50, step=50)
    lr = st.number_input("lr", value=0.05, step=0.01, format="%.3f")
    k_smooth = st.number_input("k_smooth", value=30.0, step=5.0)
    warm_start = st.checkbox("Warm start across scenarios", value=True)

    st.divider()
    st.subheader("Scenario grid (multipliers)")
    st.caption(f"Each row must have {len(MODES)} values in this order: {MODES}")
    scen_text = st.text_area(
        "One row per scenario",
        value="\n".join([
            "1.00,1.00,1.00,1.00",
            "1.10,1.05,1.00,0.95",
            "0.90,1.10,1.05,1.10",
        ]),
        height=120
    )

    run_btn = st.button("Run scenario grid", type="primary")

# -----------------------------------------------------------------------------
# Load model + data tensors
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
    st.error(f"Failed to build tensors: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Parse scenarios + build cap/mask
# -----------------------------------------------------------------------------
try:
    scenario_mult_list = parse_scenarios(scen_text, MODES)
except Exception as e:
    st.error(f"Bad scenario list: {e}")
    st.stop()

cap = torch.tensor([cap_inputs[m] for m in MODES], dtype=torch.float32)
controllable_mask = make_mask(control_modes) if control_modes else None

# -----------------------------------------------------------------------------
# Show current inputs as pretty tables
# -----------------------------------------------------------------------------
top1, top2, top3 = st.columns([1, 1, 1])

with top1:
    st.subheader("Controllable modes")
    st.dataframe(pd.DataFrame({"mode": MODES, "controllable": [m in set(control_modes) for m in MODES]}), use_container_width=True)

with top2:
    st.subheader("Capacity")
    st.dataframe(pretty_cap_table(cap), use_container_width=True)

with top3:
    st.subheader("Scenarios")
    st.dataframe(scenarios_to_df(scenario_mult_list, MODES), use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Run optimization
# -----------------------------------------------------------------------------
if run_btn:
    with st.spinner("Optimizing scenario grid..."):
        grid_df = run_scenario_grid(
            tensors=all_t,
            scaler=pipe.scaler,
            asc=pipe.model.asc,
            beta=pipe.model.beta,
            raw_la=pipe.model.raw_lam_air,
            raw_ll=pipe.model.raw_lam_land,
            cap=cap,
            scenario_mult_list=scenario_mult_list,
            controllable_mask=controllable_mask,
            steps=int(steps),
            lr=float(lr),
            k_smooth=float(k_smooth),
            warm_start=bool(warm_start),
            verbose_every=0,
        )

    grid_df_sorted = grid_df.sort_values("revenue_total", ascending=False).reset_index(drop=True)

    st.subheader("Results (all scenarios)")
    st.dataframe(grid_df_sorted.round(4), use_container_width=True)

    st.subheader("Top scenarios by total revenue")
    st.dataframe(grid_df_sorted.head(10).round(4), use_container_width=True)

    keep_cols = ["scenario_id", "revenue_total"] + [f"scenario_mult_{m}" for m in MODES] + [f"decision_mult_{m}" for m in MODES]
    st.subheader("Compact summary (multipliers + revenue)")
    st.dataframe(grid_df_sorted[keep_cols].round(4), use_container_width=True)
else:
    st.info("Set inputs on the left, then click **Run scenario grid**.")