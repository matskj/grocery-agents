from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .common import FEATURE_COLUMNS, read_table


def load_model(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score(frame: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = np.ones((x.shape[0],), dtype=float) * float(weights.get("bias", 0.0))
    for idx, name in enumerate(FEATURE_COLUMNS):
        y += x[:, idx] * float(weights.get(name, 0.0))
    return y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    err = y_true - y_pred
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained mode model.")
    parser.add_argument("--data", default="data/runs_features.parquet")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default=None)
    args = parser.parse_args()

    frame = read_table(Path(args.data))
    model = load_model(Path(args.model))
    mode = args.mode or model.get("mode")
    if mode:
        frame = frame[frame["mode"] == mode].copy()
    if frame.empty:
        raise SystemExit("no rows to evaluate")

    if "n_step_return" not in frame.columns:
        frame["n_step_return"] = frame["reward_proxy"]
    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0

    y_true = frame["n_step_return"].to_numpy(dtype=float)
    y_pred = score(frame, model.get("weights", {}))
    result = {
        "mode": mode,
        "rows": int(len(frame)),
        **compute_metrics(y_true, y_pred),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
