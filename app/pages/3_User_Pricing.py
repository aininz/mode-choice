# app/pages/3_User_Pricing.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modechoice.config import MODES
from modechoice.io import load_bundle
from modechoice.pipeline import ModeChoicePipeline
from modechoice.model import NestedLogitHetero
from modechoice.tensors import Standardizer
from modechoice.pricing_user import make_user_tensor_hetero, user_probs, optimize_user

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="User-specific Pricing", layout="wide")
st.title("🧍 User-specific Pricing")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_resource
def load_pipe(bundle_path: str) -> ModeChoicePipeline:
    return load_bundle(
        bundle_path,
        PipelineCls=ModeChoicePipeline,
        ModelCls=NestedLogitHetero,
        ScalerCls=Standardizer,
    )


def probs_dict(probs: np.ndarray) -> Dict[str, float]:
    return {m: float(probs[i]) for i, m in enumerate(MODES)}


def attrs_table(base_item: Dict[str, Dict[str, float]], item_feat_names: List[str]) -> pd.DataFrame:
    rows = []
    for m in MODES:
        feats = base_item.get(m, {})
        row = {"mode": m, "available": m in base_item}
        for f in item_feat_names:
            row[f] = float(feats.get(f, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def result_table(mult_all: np.ndarray, base_cost: np.ndarray, new_cost: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"mode": MODES})
    df["multiplier"] = [float(mult_all[i]) for i in range(len(MODES))]
    df["cost_before"] = [float(base_cost[i]) for i in range(len(MODES))]
    df["cost_after"] = [float(new_cost[i]) for i in range(len(MODES))]
    df["prob_after"] = [float(probs[i]) for i in range(len(MODES))]
    return df

def summarize_max_policy(
    probs: np.ndarray,
    avail_user_t,                 # torch tensor (1, I) or (I,)
    X_user_orig_t,                # torch tensor (1, I, D) or numpy
    cost_idx: int,
    modes: list,
    target_modes: list,
):
    # availability mask
    if hasattr(avail_user_t, "detach"):
        avail = avail_user_t.detach().cpu().numpy()
    else:
        avail = np.asarray(avail_user_t)

    avail = avail.reshape(-1)  # (I,)
    avail_mask = (avail > 0.5)

    probs = np.asarray(probs).reshape(-1)  # (I,)
    probs_av = probs.copy()
    probs_av[~avail_mask] = -np.inf  # ignore unavailable

    # max over all available modes
    i_all = int(np.argmax(probs_av))
    p_all = float(probs_av[i_all])
    mode_all = modes[i_all]

    # max over available target modes
    target_set = set(target_modes or [])
    target_mask = np.array([(m in target_set) for m in modes], dtype=bool) & avail_mask

    if target_mask.any():
        probs_t = probs.copy()
        probs_t[~target_mask] = -np.inf
        i_t = int(np.argmax(probs_t))
        p_t = float(probs_t[i_t])
        mode_t = modes[i_t]
    else:
        i_t, p_t, mode_t = None, float("-inf"), None

    # compare who wins
    winner_is_target = (mode_all in target_set)

    # revenue = 0 if winner is not target, else revenue of the max-prob target mode
    if winner_is_target and mode_t is not None and np.isfinite(p_t):
        if hasattr(X_user_orig_t, "detach"):
            Xo = X_user_orig_t.detach().cpu().numpy()
        else:
            Xo = np.asarray(X_user_orig_t)

        # Xo expected shape (1, I, D)
        price_t = float(Xo[0, i_t, cost_idx])
        revenue_rule = price_t
    else:
        revenue_rule = 0.0

    return {
        "max_all_mode": mode_all,
        "max_all_prob": p_all,
        "max_target_mode": mode_t,
        "max_target_prob": (None if not np.isfinite(p_t) else p_t),
        "which_bigger": (
            "NON-TARGET"
            if (mode_t is None or not np.isfinite(p_t))
            else (
                "TARGET"
                if (mode_all in set(target_modes))
                else (
                    "TARGET" if (p_t > p_all) else ("NON-TARGET" if (p_all > p_t) else "TIE")
                )
            )
        ),
        "winner_is_target": winner_is_target,
        "revenue_rule": revenue_rule,
        "i_all": i_all,
        "i_t": i_t,
    }

def choose_policy_with_guardrail(before_summary: dict, after_summary: dict) -> tuple[str, str | None]:
    """
    Returns (chosen_version, reason)
      - chosen_version: "before" or "after"
      - reason: human-readable message (or None)
    Guardrails:
      1) If baseline already chooses a target mode -> keep baseline (no-op).
      2) If optimization reduces best-target probability vs baseline -> revert to baseline.
    """
    # baseline already target winner -> no need to touch prices
    if before_summary.get("winner_is_target", False):
        return "before", "Baseline already picks a TARGET mode. No price change needed."

    # guardrail to not accept worse target probability
    b = before_summary.get("max_target_prob", None)
    a = after_summary.get("max_target_prob", None)

    if (b is not None) and (a is not None) and np.isfinite(b) and np.isfinite(a) and (a < b):
        return "before", "Guardrail triggered: optimization reduced best TARGET probability. Reverting to baseline."

    return "after", None

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    # st.header("Inputs")

    default_bundle = ROOT / "runtime_assets" / "modechoice_bundle"
    # BUNDLE_PATH = st.text_input("Bundle path", value=str(default_bundle))
    BUNDLE_PATH = default_bundle

    # st.divider()
    st.subheader("User profile")
    income = st.number_input("income", value=70.0, step=1.0)
    urban = st.selectbox("urban", options=[0, 1], index=1)  # clean 0/1 UI

    st.divider()
    st.subheader("Which modes are in this menu?")
    st.caption("Unchecked modes are treated as unavailable for this user query.")
    menu_modes = st.multiselect(
        "Available modes",
        options=MODES,
        default=[m for m in ["train", "car", "bus", "air"] if m in MODES] or [MODES[0]],
    )

    st.divider()
    st.subheader("Optimization")
    control_modes = st.multiselect(
        "Controllable modes (you can price)",
        options=MODES,
        default=[m for m in ["train", "air"] if m in MODES] or [MODES[0]],
    )
    target_modes = st.multiselect(
        "Target modes (1 - 3)",
        options=MODES,
        default=[m for m in ["train", "air"] if m in MODES] or [MODES[0]],
    )
    objective = st.selectbox("Objective", options=["revenue", "prob"], index=0)
    forbid_modes = st.multiselect(
        "Forbidden target modes",
        options=MODES,
        default=[m for m in ["car"] if m in MODES],
    )

    lo = st.number_input("Multiplier low", value=0.8, step=0.05)
    hi = st.number_input("Multiplier high", value=1.2, step=0.05)
    steps = st.number_input("steps", value=250, min_value=50, step=50)
    lr = st.number_input("lr", value=0.08, step=0.01, format="%.3f")


# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
try:
    pipe = load_pipe(BUNDLE_PATH)
except Exception as e:
    st.error(f"Failed to load bundle: {e}")
    st.stop()

# IMPORTANT: must use same item feats as training
item_feat_names = pipe.meta.get("item_feat_names", ["cost", "ivt", "ovt", "freq"])
item_feat_names = list(item_feat_names)

st.caption(f"Loaded model lambdas: {pipe.model.lambdas()}")
# st.caption(f"Item features expected: {item_feat_names}")

# -----------------------------------------------------------------------------
# Mode attribute inputs (in main area, nicer than cramming sidebar)
# -----------------------------------------------------------------------------
st.subheader("Menu attributes (per mode)")
st.caption("These are the attributes for this single user query (one OD / search).")

default_vals = {
    "train": {"cost": 55.0, "ivt": 60.0, "ovt": 10.0, "freq": 4.0},
    "car":   {"cost": 40.0, "ivt": 75.0, "ovt": 0.0,  "freq": 0.0},
    "bus":   {"cost": 25.0, "ivt": 95.0, "ovt": 15.0, "freq": 8.0},
    "air":   {"cost": 160.0,"ivt": 50.0, "ovt": 40.0, "freq": 3.0},
}

base_item: Dict[str, Dict[str, float]] = {}

tabs = st.tabs([m for m in MODES])  # one tab per mode
for tab, m in zip(tabs, MODES):
    with tab:
        enabled = m in set(menu_modes)
        st.checkbox(f"{m} available", value=enabled, key=f"avail_{m}", disabled=True)

        if not enabled:
            st.info(f"{m} is not in the menu. Enable it in the sidebar.")
            continue

        cols = st.columns(2)
        feats = {}
        for idx, f in enumerate(item_feat_names):
            c = cols[idx % 2]
            with c:
                dv = float(default_vals.get(m, {}).get(f, 0.0))
                feats[f] = float(
                    st.number_input(
                        f"{m} {f}",
                        value=dv,
                        step=1.0 if f != "freq" else 0.5,
                        format="%.3f" if f == "freq" else "%.2f",
                        key=f"feat_{m}_{f}",
                    )
                )
        base_item[m] = feats

st.subheader("Input summary")
c1, c2 = st.columns([1, 2])

with c1:
    user_df = pd.DataFrame([{"income": float(income), "urban": int(urban)}])
    st.markdown("**User profile**")
    st.dataframe(user_df, use_container_width=True)

with c2:
    st.markdown("**Menu attributes**")
    st.dataframe(attrs_table(base_item, item_feat_names).round(3), use_container_width=True)

# -----------------------------------------------------------------------------
# Build user tensors + baseline probs
# -----------------------------------------------------------------------------
case_features = {"income": float(income), "urban": float(urban)}

try:
    X_user_std_t, avail_user_t, X_user_orig_t = make_user_tensor_hetero(
        base_item=base_item,
        item_feat_names=item_feat_names,
        case_features=case_features,
        scaler=pipe.scaler,
    )
except Exception as e:
    st.error(f"Failed to build user tensors: {e}")
    st.stop()

baseline = user_probs(
    X_user_std_t,
    avail_user_t,
    pipe.model.asc,
    pipe.model.beta,
    pipe.model.raw_lam_air,
    pipe.model.raw_lam_land,
)
baseline = np.asarray(baseline, dtype=float)

st.subheader("Baseline choice probabilities")
base_prob_df = pd.DataFrame({"mode": MODES, "prob_baseline": baseline})
st.dataframe(base_prob_df.round(4), use_container_width=True)

# -----------------------------------------------------------------------------
# Optimize
# -----------------------------------------------------------------------------
run_opt = st.button("Optimize user pricing", type="primary")

if run_opt:
    cost_k = pipe.scaler.feat_names.index("cost")
    # baseline summary first so we can no-op
    before_summary = summarize_max_policy(
        probs=baseline,
        avail_user_t=avail_user_t,
        X_user_orig_t=X_user_orig_t,
        cost_idx=cost_k,
        modes=MODES,
        target_modes=target_modes,
    )

    # if baseline already wins with a target, then no need to optimize
    if before_summary["winner_is_target"]:
        chosen_version, reason = "before", "Baseline already picks a TARGET mode. No price change needed."

        # "after" placeholders = baseline (so downstream code is simple)
        mult_all = np.ones(len(MODES), dtype=float)
        base_cost = X_user_orig_t.detach().cpu().numpy()[0, :, cost_k]
        new_cost = base_cost.copy()
        probs = baseline.copy()

        after_summary = before_summary

    else:
        # run optimization normally
        try:
            mult_all, base_cost, new_cost, probs = optimize_user(
                X_user_orig_t=X_user_orig_t,
                avail_user_t=avail_user_t,
                scaler=pipe.scaler,
                asc=pipe.model.asc,
                beta=pipe.model.beta,
                raw_la=pipe.model.raw_lam_air,
                raw_ll=pipe.model.raw_lam_land,
                control_modes=control_modes,
                target_modes=target_modes,
                objective=objective,
                mult_bounds=(float(lo), float(hi)),
                steps=int(steps),
                lr=float(lr),
                forbid_modes=forbid_modes,
            )
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            st.stop()

        probs = np.asarray(probs, dtype=float)

        after_summary = summarize_max_policy(
            probs=probs,
            avail_user_t=avail_user_t,
            X_user_orig_t=X_user_orig_t,
            cost_idx=cost_k,
            modes=MODES,
            target_modes=target_modes,
        )

        # guardrail decision
        chosen_version, reason = choose_policy_with_guardrail(before_summary, after_summary)

    # choose what to apply (before vs after)
    if chosen_version == "after":
        applied_mult = mult_all
        applied_cost = new_cost
        applied_probs = probs
        applied_summary = after_summary
    else:
        applied_mult = np.ones(len(MODES), dtype=float)
        applied_cost = base_cost
        applied_probs = baseline
        applied_summary = before_summary

    if reason:
        st.info(f"Applied policy: **{chosen_version.upper()}**. {reason}")

    # show optimization results (unchanged, optional)
    st.subheader("Optimization results")
    res_df = result_table(mult_all, base_cost, new_cost, probs)
    res_df = res_df.merge(pd.DataFrame({"mode": MODES, "prob_baseline": baseline}), on="mode", how="left")
    res_df["prob_delta"] = res_df["prob_after"] - res_df["prob_baseline"]
    res_df = res_df[["mode","multiplier","cost_before","cost_after","prob_baseline","prob_after","prob_delta"]]
    st.dataframe(res_df.round(4), use_container_width=True)

    # show APPLIED results (this is the key UX fix)
    st.subheader("Applied pricing (post-guardrail)")
    applied_df = pd.DataFrame({
        "mode": MODES,
        "multiplier_applied": [float(applied_mult[i]) for i in range(len(MODES))],
        "cost_applied": [float(applied_cost[i]) for i in range(len(MODES))],
        "prob_applied": [float(applied_probs[i]) for i in range(len(MODES))],
    })
    st.dataframe(applied_df.round(4), use_container_width=True)

    # applied policy summary
    st.subheader("Max-probability policy summary (APPLIED)")

    winner_label = applied_summary["which_bigger"]
    if chosen_version == "before" and before_summary["winner_is_target"]:
        winner_label = "TARGET (no change)"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max P among ALL modes", f"{applied_summary['max_all_prob']:.4f}", applied_summary["max_all_mode"])
    c2.metric(
        "Max P among TARGET modes",
        "—" if applied_summary["max_target_mode"] is None else f"{applied_summary['max_target_prob']:.4f}",
        "none" if applied_summary["max_target_mode"] is None else applied_summary["max_target_mode"],
    )
    c3.metric("Winner", winner_label)
    c4.metric("Expected Revenue", f"{applied_summary['revenue_rule']:.2f}",
              delta=f"{applied_summary['revenue_rule'] - before_summary['revenue_rule']:.2f}")