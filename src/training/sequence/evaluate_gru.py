"""
evaluate_gru.py

Evaluates the trained GRU on the held-out test-satellite windows, inverse-transforms
predictions with expm1, applies the exact same bucketed MAE analysis as
src/training/evaluation/01_bucketed_error_analysis.py, and prints a side-by-side
comparison against Variant D HistGradientBoosting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

SEQ_SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SEQ_SRC_DIR.parent.parent.parent
SEQ_DATA_DIR = BASE_DIR / "data" / "sequence"
MODEL_PATH = SEQ_SRC_DIR / "gru_model.pt"
META_PATH = SEQ_SRC_DIR / "gru_meta.npz"

sys.path.insert(0, str(SEQ_SRC_DIR))
from train_gru import SatelliteGRU

# Exact same decay horizon bins as src/training/evaluation/01_bucketed_error_analysis.py
BUCKET_BINS = [0, 30, 180, 365, np.inf]
BUCKET_LABELS = ["<30 days", "30-180 days", "180-365 days", ">365 days"]

# Known Variant D HGB test metrics (from README / 03_evaluate_improvements.py)
VARIANT_D_METRICS = {
    "ALL (Overall)": {"MAE": 231.32, "RMSE": 605.02},
    "<30 days": {"MAE": 2.44},
    "30-180 days": {"MAE": 13.15},
    "180-365 days": {"MAE": 33.86},
    ">365 days": {"MAE": 300.28},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained GRU against Variant D HGB.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Inference batch size")
    return parser.parse_args()


def compare_verdict(gru_mae: float, var_d_mae: float, tol: float = 0.05) -> str:
    diff = gru_mae - var_d_mae
    if abs(diff) <= tol:
        return f"MATCHED (diff={diff:+.2f}d)"
    if diff < 0:
        return f"BEAT Variant D by {-diff:.2f}d"
    return f"LOST to Variant D by +{diff:.2f}d"


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_test_path = SEQ_DATA_DIR / "test_windows.npy"
    y_test_path = SEQ_DATA_DIR / "test_targets.npy"
    if not (X_test_path.exists() and y_test_path.exists()):
        raise FileNotFoundError(f"Test window files not found in {SEQ_DATA_DIR}. Run build_windows.py first.")
    if not (MODEL_PATH.exists() and META_PATH.exists()):
        raise FileNotFoundError(f"Trained model/meta not found in {SEQ_SRC_DIR}. Run train_gru.py first.")

    X_test = np.load(X_test_path)
    y_test_log = np.load(y_test_path)

    meta = np.load(META_PATH)
    feat_mean = meta["feat_mean"]
    feat_std = meta["feat_std"]
    input_size = int(meta["input_size"])
    hidden_size = int(meta["hidden_size"])
    num_layers = int(meta["num_layers"])

    X_test_scaled = ((X_test - feat_mean) / feat_std).astype(np.float32)

    model = SatelliteGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test_scaled)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    preds_log_list: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            preds_log_list.append(model(xb).cpu().numpy())

    preds_log = np.concatenate(preds_log_list, axis=0)
    y_pred = np.clip(np.expm1(preds_log), 0.0, None)
    y_true = np.clip(np.expm1(y_test_log), 0.0, None)

    eval_df = pd.DataFrame({"days_to_decay": y_true, "pred": y_pred})
    eval_df["bucket"] = pd.cut(eval_df["days_to_decay"], bins=BUCKET_BINS, labels=BUCKET_LABELS, right=False)

    # Compute GRU overall + bucketed metrics
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = root_mean_squared_error(y_true, y_pred)
    overall_mean_act = float(y_true.mean())
    overall_rel_err = (overall_mae / overall_mean_act) * 100.0

    gru_rows = [
        {
            "Bucket": "ALL (Overall)",
            "Observations": len(eval_df),
            "Share (%)": 100.0,
            "Mean Actual (d)": round(overall_mean_act, 2),
            "GRU MAE (d)": round(overall_mae, 2),
            "GRU RMSE (d)": round(overall_rmse, 2),
            "GRU Rel Err (%)": round(overall_rel_err, 1),
            "Variant D MAE (d)": VARIANT_D_METRICS["ALL (Overall)"]["MAE"],
            "Variant D RMSE (d)": VARIANT_D_METRICS["ALL (Overall)"]["RMSE"],
            "Verdict": compare_verdict(overall_mae, VARIANT_D_METRICS["ALL (Overall)"]["MAE"]),
        }
    ]

    for b in BUCKET_LABELS:
        sub = eval_df[eval_df["bucket"] == b]
        if len(sub) == 0:
            gru_rows.append(
                {
                    "Bucket": b,
                    "Observations": 0,
                    "Share (%)": 0.0,
                    "Mean Actual (d)": np.nan,
                    "GRU MAE (d)": np.nan,
                    "GRU RMSE (d)": np.nan,
                    "GRU Rel Err (%)": np.nan,
                    "Variant D MAE (d)": VARIANT_D_METRICS[b]["MAE"],
                    "Variant D RMSE (d)": np.nan,
                    "Verdict": "N/A (0 observations)",
                }
            )
            continue

        act = sub["days_to_decay"]
        pred = sub["pred"]
        mae = mean_absolute_error(act, pred)
        rmse = root_mean_squared_error(act, pred)
        mean_act = float(act.mean())
        rel_err = (mae / mean_act * 100.0) if mean_act > 0 else np.nan

        gru_rows.append(
            {
                "Bucket": b,
                "Observations": len(sub),
                "Share (%)": round(len(sub) / len(eval_df) * 100.0, 2),
                "Mean Actual (d)": round(mean_act, 2),
                "GRU MAE (d)": round(mae, 2),
                "GRU RMSE (d)": round(rmse, 2),
                "GRU Rel Err (%)": round(rel_err, 1),
                "Variant D MAE (d)": VARIANT_D_METRICS[b]["MAE"],
                "Variant D RMSE (d)": np.nan,
                "Verdict": compare_verdict(mae, VARIANT_D_METRICS[b]["MAE"]),
            }
        )

    comp_df = pd.DataFrame(gru_rows)

    print("\n" + "=" * 100)
    print(f"GRU (hidden_size={hidden_size}, layers={num_layers}) VS. VARIANT D HGB — TEST SET COMPARISON")
    print("=" * 100)
    print(comp_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("BUCKET-BY-BUCKET SUMMARY")
    print("=" * 100)
    for row in gru_rows:
        b = row["Bucket"]
        print(f"  - {b:<14s}: GRU MAE = {row['GRU MAE (d)']}d vs Variant D MAE = {row['Variant D MAE (d)']}d -> {row['Verdict']}")


if __name__ == "__main__":
    main()
