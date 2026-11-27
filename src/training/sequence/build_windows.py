"""
build_windows.py

Builds fixed-length (W=20) sliding-window sequences of raw per-epoch orbital
parameters from data/modeling/features.parquet, strictly respecting the existing
satellite-level train/test split in data/splits/.

No pre-engineered rolling features are included: this tests whether a GRU can
learn temporal decay trends directly from raw consecutive TLE observations.

Outputs (saved to data/sequence/):
  - train_windows.npy  (N_train, W, 9) float32
  - train_targets.npy  (N_train,)      float32  (log1p of days_to_decay)
  - train_sat_ids.npy  (N_train,)      int64
  - test_windows.npy   (N_test, W, 9)  float32
  - test_targets.npy   (N_test,)       float32  (log1p of days_to_decay)
  - test_sat_ids.npy   (N_test,)       int64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "modeling" / "features.parquet"
SPLITS_DIR = BASE_DIR / "data" / "splits"
TRAIN_SATS_CSV = SPLITS_DIR / "train_satellites.csv"
TEST_SATS_CSV = SPLITS_DIR / "test_satellites.csv"
SEQ_DATA_DIR = BASE_DIR / "data" / "sequence"

RAW_FEATURES = [
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "PERIAPSIS",
    "APOAPSIS",
    "SEMIMAJOR_AXIS",
    "PERIOD",
]
TARGET_COL = "days_to_decay"
SAT_ID_COL = "NORAD_CAT_ID"
EPOCH_COL = "EPOCH"


def build_split_windows(
    df_split: pd.DataFrame,
    sat_ids: np.ndarray,
    window_size: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create non-overlapping-across-satellites sliding windows for one split."""
    windows_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    sat_id_list: list[np.ndarray] = []
    skipped: list[tuple[int, int]] = []

    # Group once for fast per-satellite lookup
    grouped = {int(k): v for k, v in df_split.groupby(SAT_ID_COL, sort=False)}

    for sat_id in sat_ids:
        sat_id_int = int(sat_id)
        sat_df = grouped.get(sat_id_int)
        if sat_df is None or len(sat_df) < window_size:
            n_rows = 0 if sat_df is None else len(sat_df)
            skipped.append((sat_id_int, n_rows))
            continue

        # Sort chronologically by observation epoch
        sat_df = sat_df.assign(_epoch_dt=pd.to_datetime(sat_df[EPOCH_COL], utc=True))
        sat_df = sat_df.sort_values("_epoch_dt", kind="mergesort")

        # Drop any rows with NaN/inf in raw features or target
        cols = RAW_FEATURES + [TARGET_COL]
        valid_mask = np.isfinite(sat_df[cols].to_numpy(dtype=np.float64)).all(axis=1)
        if not valid_mask.all():
            sat_df = sat_df.loc[valid_mask]

        if len(sat_df) < window_size:
            skipped.append((sat_id_int, len(sat_df)))
            continue

        X_sat = sat_df[RAW_FEATURES].to_numpy(dtype=np.float32)
        y_raw = sat_df[TARGET_COL].to_numpy(dtype=np.float64)
        y_raw = np.clip(y_raw, 0.0, None)
        y_log = np.log1p(y_raw).astype(np.float32)

        # sliding_window_view over axis 0 produces shape (n_win, n_features, window_size)
        win_view = np.lib.stride_tricks.sliding_window_view(X_sat, window_size, axis=0)
        win_sat = np.transpose(win_view, (0, 2, 1)).copy()
        tgt_sat = y_log[window_size - 1 :].copy()
        ids_sat = np.full(len(tgt_sat), sat_id_int, dtype=np.int64)

        windows_list.append(win_sat)
        targets_list.append(tgt_sat)
        sat_id_list.append(ids_sat)

    print(f"\n[{split_name.upper()}] Processed {len(sat_ids)} satellites (window_size={window_size}):")
    print(f"  Skipped (< {window_size} valid epochs): {len(skipped)} satellites")
    if skipped:
        skipped_str = ", ".join(f"{sid} ({n} epochs)" for sid, n in skipped)
        print(f"  Skipped IDs: {skipped_str}")

    if not windows_list:
        raise RuntimeError(f"No valid windows constructed for {split_name} split.")

    X_out = np.concatenate(windows_list, axis=0)
    y_out = np.concatenate(targets_list, axis=0)
    ids_out = np.concatenate(sat_id_list, axis=0)
    print(f"  Built {len(X_out):,} windows | shape={X_out.shape} | targets={y_out.shape}")
    return X_out, y_out, ids_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw sliding-window sequences for GRU training.")
    parser.add_argument("--window-size", type=int, default=20, help="Consecutive epochs per window (default: 20)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_size = args.window_size

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Modeling dataset not found: {FEATURES_PATH}")
    if not TRAIN_SATS_CSV.exists() or not TEST_SATS_CSV.exists():
        raise FileNotFoundError(f"Split files not found in {SPLITS_DIR}")

    train_sats = pd.read_csv(TRAIN_SATS_CSV)[SAT_ID_COL].to_numpy(dtype=np.int64)
    test_sats = pd.read_csv(TEST_SATS_CSV)[SAT_ID_COL].to_numpy(dtype=np.int64)

    overlap = set(train_sats).intersection(set(test_sats))
    if overlap:
        raise ValueError(f"Satellite overlap detected between train and test splits: {overlap}")

    print(f"Loading raw columns from {FEATURES_PATH}...")
    read_cols = [SAT_ID_COL, EPOCH_COL, TARGET_COL] + RAW_FEATURES
    df = pd.read_parquet(FEATURES_PATH, columns=read_cols)
    print(f"Loaded {len(df):,} rows across {df[SAT_ID_COL].nunique()} satellites.")
    print(f"Using {len(RAW_FEATURES)} raw per-epoch features: {RAW_FEATURES}")

    train_df = df[df[SAT_ID_COL].isin(train_sats)].copy()
    test_df = df[df[SAT_ID_COL].isin(test_sats)].copy()

    X_train, y_train, train_ids = build_split_windows(train_df, train_sats, window_size, "train")
    X_test, y_test, test_ids = build_split_windows(test_df, test_sats, window_size, "test")

    SEQ_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SEQ_DATA_DIR / "train_windows.npy", X_train)
    np.save(SEQ_DATA_DIR / "train_targets.npy", y_train)
    np.save(SEQ_DATA_DIR / "train_sat_ids.npy", train_ids)
    np.save(SEQ_DATA_DIR / "test_windows.npy", X_test)
    np.save(SEQ_DATA_DIR / "test_targets.npy", y_test)
    np.save(SEQ_DATA_DIR / "test_sat_ids.npy", test_ids)

    print(f"\nSaved windowed arrays to {SEQ_DATA_DIR}")


if __name__ == "__main__":
    main()
