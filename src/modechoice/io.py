from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type

import torch


def _scaler_to_dict(scaler) -> Dict[str, Any]:
    if is_dataclass(scaler):
        d = asdict(scaler)
    else:
        d = {
            "mu": getattr(scaler, "mu"),
            "sd": getattr(scaler, "sd"),
            "feat_names": getattr(scaler, "feat_names"),
        }

    d["mu"] = [float(x) for x in d["mu"]]
    d["sd"] = [float(x) for x in d["sd"]]
    d["feat_names"] = list(d["feat_names"])
    return d


def save_bundle(path: str | Path, pipe) -> Path:
    """
    Save a DIRECTORY bundle at `path/` containing:
      - model.pt   (torch tensors only)
      - scaler.json
      - meta.json

    NOTE: `path` must be a DIRECTORY path (no .pt suffix).
    """
    path = Path(path)

    if path.suffix in {".pt", ".pth"}:
        raise ValueError(
            f"save_bundle expects a DIRECTORY path (no .pt/.pth). Got: {path}"
        )

    path.mkdir(parents=True, exist_ok=True)

    model_state = {
        "asc": pipe.model.asc.detach().cpu(),
        "beta": pipe.model.beta.detach().cpu(),
        "raw_lam_air": pipe.model.raw_lam_air.detach().cpu(),
        "raw_lam_land": pipe.model.raw_lam_land.detach().cpu(),
    }
    torch.save(model_state, path / "model.pt")

    scaler_dict = _scaler_to_dict(pipe.scaler)
    (path / "scaler.json").write_text(json.dumps(scaler_dict, indent=2), encoding="utf-8")

    meta = pipe.meta if getattr(pipe, "meta", None) is not None else {}
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return path


def load_bundle(
    path: str | Path,
    PipelineCls: Type,
    ModelCls: Type,
    ScalerCls: Type,
):
    """
    Load a DIRECTORY bundle created by save_bundle().
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {path}")

    model_path = path / "model.pt"
    scaler_path = path / "scaler.json"
    meta_path = path / "meta.json"

    state = torch.load(model_path, map_location="cpu", weights_only=True)

    scaler_d = json.loads(scaler_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    model = ModelCls(state["asc"], state["beta"], state["raw_lam_air"], state["raw_lam_land"])
    scaler = ScalerCls(**scaler_d)

    return PipelineCls(model=model, scaler=scaler, meta=meta)