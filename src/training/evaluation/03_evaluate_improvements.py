"""
03_evaluate_improvements.py

Systematic evaluation of modeling improvements against the baseline:
  - Preserves the exact satellite-level train/test split.
  - Tests three distinct hypotheses independently and in combination:
      1. Variant A (Temporal Progression): Adds days_since_first_epoch.
      2. Variant B (Short Rolling Window): Adds 3-epoch rolling features (mean_motion_roll3, etc.).
      3. Variant C (Target Log-Transform): Trains on log1p(days_to_decay) and inverse-transforms with expm1.
      4. Variant D (Combined A + B + C): Full synergy of temporal features, short window, and log loss.
  - Evaluates both OVERALL MAE/RMSE and the 4 BUCKETED DECAY HORIZONS:
      - <30 days
      - 30-180 days
      - 180-365 days
      - >365 days
  - Saves updated model artifacts to models/.
  - Saves comparison tables to reports/modeling/model_improvements_comparison.csv.
  - Saves predicted-vs-actual plots for the top improvements.

Run via:
  python src/training/evaluation/03_evaluate_improvements.py
"""

from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Add src/training to import data_loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import FEATURE_COLS, TARGET_COL, save_scatter_plot

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "modeling" / "features.parquet"
SPLITS_DIR = BASE_DIR / "data" / "splits"
TEST_SATS_CSV = SPLITS_DIR / "test_satellites.csv"
TRAIN_SATS_CSV = SPLITS_DIR / "train_satellites.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "modeling"
OUTPUT_CSV = REPORTS_DIR / "model_improvements_comparison.csv"

# Decay horizon bins
BUCKET_BINS = [0, 30, 180, 365, np.inf]
BUCKET_LABELS = ["<30 days", "30-180 days", "180-365 days", ">365 days"]


def prepare_augmented_dataset():
    """
    Loads features.parquet and engineers the improvement candidates:
      - days_since_first_epoch (elapsed physical tracking time)
      - 3-epoch short rolling windows for orbital features
    Applies strict satellite-level partitioning with zero overlap.
    """
    print("=" * 80)
    print("ENGINEERING CANDIDATE FEATURES & PREPARING DATASETS")
    print("=" * 80)

    df = pd.read_parquet(FEATURES_PATH)
    test_sats = pd.read_csv(TEST_SATS_CSV)["NORAD_CAT_ID"].values
    train_sats = pd.read_csv(TRAIN_SATS_CSV)["NORAD_CAT_ID"].values

    # 1. Feature A: days_since_first_epoch (physical time elapsed along tracking arc)
    df["_epoch_dt"] = pd.to_datetime(df["EPOCH"], utc=True)
    df["_first_epoch"] = df.groupby("NORAD_CAT_ID")["_epoch_dt"].transform("min")
    df["days_since_first_epoch"] = (df["_epoch_dt"] - df["_first_epoch"]).dt.total_seconds() / 86400.0
    df = df.drop(columns=["_epoch_dt", "_first_epoch"])

    # 2. Feature B: Short rolling window (3 epochs)
    for param in ["MEAN_MOTION", "BSTAR", "PERIAPSIS", "ECCENTRICITY"]:
        col_name = f"{param.lower()}_roll3"
        df[col_name] = (
            df.groupby("NORAD_CAT_ID")[param]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )

    # 3. Handle boundary NaNs (drop 195 initial boundary rows)
    diff_cols = ["mean_motion_rate", "bstar_rate", "eccentricity_rate", "periapsis_rate"]
    nan_mask = df[diff_cols].isna().any(axis=1)
    df_clean = df[~nan_mask].copy()

    # 4. Strict satellite split
    train_df = df_clean[df_clean["NORAD_CAT_ID"].isin(train_sats)].copy()
    test_df = df_clean[df_clean["NORAD_CAT_ID"].isin(test_sats)].copy()

    # Verify zero overlap
    overlap = set(train_df["NORAD_CAT_ID"]).intersection(set(test_df["NORAD_CAT_ID"]))
    assert len(overlap) == 0, f"Critical overlap: {overlap}"

    # Assign buckets to test set
    test_df["bucket"] = pd.cut(test_df[TARGET_COL], bins=BUCKET_BINS, labels=BUCKET_LABELS, right=False)

    print(f"  Train observations: {len(train_df):,} | Test observations: {len(test_df):,}")
    print(f"  ✓ Candidate features computed successfully.")

    return train_df, test_df


def evaluate_candidate(
    variant_name,
    feature_list,
    train_df,
    test_df,
    log_transform_target=False,
    model_save_path=None,
    plot_save_path=None,
):
    """
    Trains a HistGradientBoosting model under a specific variant configuration
    and computes overall + bucketed metrics.
    """
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {variant_name.upper()}")
    print(f"  Features ({len(feature_list)}): {feature_list[:4]} ... + {len(feature_list)-4} more")
    print(f"  Target Mode: {'log1p(y) -> expm1(pred)' if log_transform_target else 'Raw days_to_decay'}")
    print("=" * 80)

    X_train = train_df[feature_list]
    X_test = test_df[feature_list]

    if log_transform_target:
        y_train = np.log1p(train_df[TARGET_COL])
    else:
        y_train = train_df[TARGET_COL]

    y_test = test_df[TARGET_COL]

    # Train model
    model = HistGradientBoostingRegressor(random_state=42)
    t_start = time.time()
    model.fit(X_train, y_train)
    wall_clock = time.time() - t_start

    # Predict & transform
    preds = model.predict(X_test)
    if log_transform_target:
        preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)  # Clip negative predictions to 0

    test_df["current_pred"] = preds

    # Overall metrics
    overall_mae = mean_absolute_error(y_test, preds)
    overall_rmse = root_mean_squared_error(y_test, preds)
    overall_mean_act = y_test.mean()

    print(f"  ✓ Wall-clock time: {wall_clock:.2f}s")
    print(f"  ✓ Overall Performance: MAE = {overall_mae:.2f} days | RMSE = {overall_rmse:.2f} days")

    # Bucketed metrics
    bucket_records = []
    print("\n  Horizon Performance Breakdown:")
    print(f"    {'Horizon':<15s} {'MAE (d)':<10s} {'RMSE (d)':<10s} {'Mean Act (d)':<14s} {'Rel Error (%)':<12s}")
    print("    " + "-" * 62)

    for b in BUCKET_LABELS:
        sub = test_df[test_df["bucket"] == b]
        act = sub[TARGET_COL]
        p = sub["current_pred"]

        mae = mean_absolute_error(act, p)
        rmse = root_mean_squared_error(act, p)
        mean_act = act.mean()
        rel_err = (mae / mean_act * 100) if mean_act > 0 else np.nan

        print(f"    {b:<15s} {mae:>8.2f}d  {rmse:>8.2f}d  {mean_act:>10.2f}d  {rel_err:>10.1f}%")

        bucket_records.append({
            "Variant": variant_name,
            "Bucket": b,
            "Observations": len(sub),
            "Mean Actual (d)": round(mean_act, 2),
            "MAE (d)": round(mae, 2),
            "RMSE (d)": round(rmse, 2),
            "Relative Error (%)": round(rel_err, 1),
        })

    # Save model artifact if path provided
    if model_save_path:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_save_path)
        print(f"\n  ✓ Saved model artifact to: {model_save_path}")

    # Save plot if path provided
    if plot_save_path:
        save_scatter_plot(
            y_test=y_test,
            y_pred=preds,
            model_name=f"HGB ({variant_name})",
            output_paths=[plot_save_path],
            metrics={"MAE": overall_mae, "RMSE": overall_rmse},
        )

    summary_entry = {
        "Variant": variant_name,
        "Overall MAE (d)": round(overall_mae, 2),
        "Overall RMSE (d)": round(overall_rmse, 2),
        "<30d MAE (d)": bucket_records[0]["MAE (d)"],
        "<30d RelErr (%)": bucket_records[0]["Relative Error (%)"],
        "30-180d MAE (d)": bucket_records[1]["MAE (d)"],
        "30-180d RelErr (%)": bucket_records[1]["Relative Error (%)"],
        "180-365d MAE (d)": bucket_records[2]["MAE (d)"],
        ">365d MAE (d)": bucket_records[3]["MAE (d)"],
        "Training Time (s)": round(wall_clock, 2),
    }

    return summary_entry, bucket_records


def main():
    train_df, test_df = prepare_augmented_dataset()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Define feature sets
    short_roll_cols = ["mean_motion_roll3", "bstar_roll3", "periapsis_roll3", "eccentricity_roll3"]
    temporal_cols = ["days_since_first_epoch"]

    variants = [
        {
            "name": "Baseline (Raw Target)",
            "features": FEATURE_COLS,
            "log_target": False,
            "model_path": None,  # Already exists
            "plot_path": None,
        },
        {
            "name": "Variant A: +days_since_first_epoch",
            "features": FEATURE_COLS + temporal_cols,
            "log_target": False,
            "model_path": MODELS_DIR / "hgb_variant_a_days_elapsed.joblib",
            "plot_path": REPORTS_DIR / "predicted_vs_actual_variant_a_elapsed.png",
        },
        {
            "name": "Variant B: +Short Window (roll3)",
            "features": FEATURE_COLS + short_roll_cols,
            "log_target": False,
            "model_path": MODELS_DIR / "hgb_variant_b_short_window.joblib",
            "plot_path": REPORTS_DIR / "predicted_vs_actual_variant_b_short.png",
        },
        {
            "name": "Variant C: Log Target log1p(y)",
            "features": FEATURE_COLS,
            "log_target": True,
            "model_path": MODELS_DIR / "hgb_variant_c_log_target.joblib",
            "plot_path": REPORTS_DIR / "predicted_vs_actual_variant_c_log.png",
        },
        {
            "name": "Variant D: Combined (A + B + C)",
            "features": FEATURE_COLS + temporal_cols + short_roll_cols,
            "log_target": True,
            "model_path": MODELS_DIR / "hgb_variant_d_combined.joblib",
            "plot_path": REPORTS_DIR / "predicted_vs_actual_variant_d_combined.png",
        },
    ]

    summary_list = []
    all_bucket_records = []

    for v in variants:
        summary, b_records = evaluate_candidate(
            variant_name=v["name"],
            feature_list=v["features"],
            train_df=train_df,
            test_df=test_df,
            log_transform_target=v["log_target"],
            model_save_path=v["model_path"],
            plot_save_path=v["plot_path"],
        )
        summary_list.append(summary)
        all_bucket_records.extend(b_records)

    # Print summary comparison table
    print("\n" + "=" * 90)
    print("SYSTEMATIC COMPARISON: BASELINE VS. IMPROVEMENTS")
    print("=" * 90)
    summary_df = pd.DataFrame(summary_list)
    print(summary_df.to_string(index=False))

    # Export comparison CSV
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved comparison report to: {OUTPUT_CSV}")

    detailed_csv = REPORTS_DIR / "model_improvements_bucket_detail.csv"
    pd.DataFrame(all_bucket_records).to_csv(detailed_csv, index=False)
    print(f"✓ Saved detailed bucketed report to: {detailed_csv}")


if __name__ == "__main__":
    main()
