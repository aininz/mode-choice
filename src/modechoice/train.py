from __future__ import annotations
from typing import Tuple
import torch
from .model import nested_logit_log_probs

def mean_nll(tensors, asc, beta, raw_la, raw_ll):
    logP = nested_logit_log_probs(tensors["X_item"], tensors["avail"], asc, beta, raw_la, raw_ll)
    y = tensors["y"]
    return -(logP[torch.arange(len(y)), y]).mean()

def fit_nested_logit(
    train_t,
    val_t,
    lr=0.05,
    max_epochs=800,
    patience=25,
    seed=0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    D = train_t["X_item"].shape[2]
    asc = torch.zeros(4, requires_grad=True)
    beta = torch.zeros(D, requires_grad=True)
    raw_la = torch.tensor(0.0, requires_grad=True)
    raw_ll = torch.tensor(0.0, requires_grad=True)

    opt = torch.optim.Adam([asc, beta, raw_la, raw_ll], lr=lr)

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(max_epochs):
        opt.zero_grad()
        loss = mean_nll(train_t, asc, beta, raw_la, raw_ll)
        loss.backward()
        opt.step()

        with torch.no_grad():
            v = mean_nll(val_t, asc, beta, raw_la, raw_ll).item()

        if v < best_val - 1e-5:
            best_val = v
            best_state = (asc.detach().clone(), beta.detach().clone(), raw_la.detach().clone(), raw_ll.detach().clone())
            bad = 0
        else:
            bad += 1

        if epoch % 100 == 0:
            print(f"epoch {epoch:3d} | train NLL={loss.item():.4f} | val NLL={v:.4f}")

        if bad >= patience:
            print(f"early stop @ epoch {epoch}, best val NLL={best_val:.4f}")
            break

    return best_state