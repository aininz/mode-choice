from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from .config import MODES, N_ITEMS, ID2ALT
from .model import nested_logit_log_probs

@torch.no_grad()
def _probs(tensors, asc, beta, raw_la, raw_ll):
    logP = nested_logit_log_probs(tensors["X_item"], tensors["avail"], asc, beta, raw_la, raw_ll)
    P = torch.exp(logP)
    return P / P.sum(dim=1, keepdim=True), logP

def eval_basic(name, tensors, asc, beta, raw_la, raw_ll):
    P, logP = _probs(tensors, asc, beta, raw_la, raw_ll)
    y = tensors["y"]
    nll = -(logP[torch.arange(len(y)), y]).mean().item()
    acc = (P.argmax(dim=1) == y).float().mean().item()

    pred_share = P.mean(dim=0).cpu().numpy()
    actual = pd.Series(y.cpu().numpy()).value_counts(normalize=True).sort_index()
    actual_share = np.array([actual.get(i, 0.0) for i in range(N_ITEMS)])

    print(f"\n{name}")
    print("per-case NLL:", round(nll, 4))
    print("accuracy:    ", round(acc, 4))
    print("pred share:  ", {ID2ALT[i]: float(pred_share[i]) for i in range(N_ITEMS)})
    print("actual share:", {ID2ALT[i]: float(actual_share[i]) for i in range(N_ITEMS)})

def eval_classification(name, tensors, asc, beta, raw_la, raw_ll):
    P, _ = _probs(tensors, asc, beta, raw_la, raw_ll)
    Pn = P.cpu().numpy()
    y = tensors["y"].cpu().numpy()
    yhat = Pn.argmax(axis=1)

    cm = confusion_matrix(y, yhat, labels=np.arange(N_ITEMS))
    cm_df = pd.DataFrame(cm, index=[f"true_{m}" for m in MODES], columns=[f"pred_{m}" for m in MODES])
    print(f"\n{name} — Confusion matrix")
    display(cm_df)

    print("\nClassification report")
    print(classification_report(y, yhat, target_names=MODES, digits=4))

    auc_macro = roc_auc_score(y, Pn, multi_class="ovr", average="macro")
    auc_weighted = roc_auc_score(y, Pn, multi_class="ovr", average="weighted")
    print("\nROC AUC macro:", round(float(auc_macro), 4))
    print("ROC AUC weighted:", round(float(auc_weighted), 4))

    pr_per_class = {}
    supports = {}
    for k, lab in enumerate(MODES):
        y_bin = (y == k).astype(int)
        supports[lab] = int(y_bin.sum())
        pr_per_class[lab] = np.nan if (y_bin.min() == y_bin.max()) else average_precision_score(y_bin, Pn[:, k])

    vals = [v for v in pr_per_class.values() if not np.isnan(v)]
    pr_macro = float(np.mean(vals)) if vals else np.nan
    print("\nPR AUC macro:", None if np.isnan(pr_macro) else round(float(pr_macro), 4))
    print("PR AUC per class:", {k: (None if np.isnan(v) else round(float(v), 4)) for k, v in pr_per_class.items()})
    print("supports:", supports)

def diagnostic_true_class_table(tensors, asc, beta, raw_la, raw_ll) -> pd.DataFrame:
    """
    Mean predicted probabilities conditioned on the true class + argmax recall.
    Helps you see e.g. true-train cases getting P(train) too low.
    """
    P, _ = _probs(tensors, asc, beta, raw_la, raw_ll)
    Pn = P.cpu().numpy()
    y = tensors["y"].cpu().numpy()
    yhat = Pn.argmax(axis=1)

    rows = []
    for k, lab in enumerate(MODES):
        mask = (y == k)
        if mask.sum() == 0:
            continue
        meanP = Pn[mask].mean(axis=0)
        recall = (yhat[mask] == k).mean()
        row = {"true_class": lab, "n_cases": int(mask.sum()), "recall_argmax": float(recall)}
        for j, mj in enumerate(MODES):
            row[f"meanP_{mj}"] = float(meanP[j])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_cases", ascending=False)
