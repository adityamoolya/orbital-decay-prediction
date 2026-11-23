"""
01_build_features.py

Feature extraction for satellite decay prediction.

Reads the deduped + filtered table from data_processing and produces
the final modeling-ready dataset.

Pipeline:
  1. Load deduped table (tle_features_modeling.parquet — 195 sats, ~832k rows).
  2. Cast numeric orbital columns from string → float64.
  3. Compute the label: days_to_decay = CATALOG_DECAY_DATE − EPOCH (days).
  4. Use Space-Track's native SEMIMAJOR_AXIS / APOAPSIS / PERIAPSIS / PERIOD
     directly — no re-derivation needed.
  5. Compute per-satellite rolling/trend features:
       - MEAN_MOTION rate of change (rev/day per day)
       - BSTAR trend (rolling mean, rate of change)
       - ECCENTRICITY rate of change
       - PERIAPSIS rate of change (altitude decay rate)
  6. Handle negative days_to_decay:
       small negatives (≤ 1 day) → clip to 0 (tracking-latency noise),
       large negatives → flag for review.
  7. Drop columns not useful for modeling.
  8. Export to data/modeling/.

Input:  data/preprocessed/tle_features_modeling.parquet
Output: data/modeling/features.parquet
        data/modeling/features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data" / "curated" / "tle_features_modeling.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "modeling"
OUTPUT_PARQUET = OUTPUT_DIR / "features.parquet"
OUTPUT_CSV = OUTPUT_DIR / "features.csv"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
SECONDS_PER_DAY = 86400
NEGATIVE_CLIP_THRESHOLD_DAYS = 1.0

# Columns to cast from string → float64
NUMERIC_COLUMNS = [
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS",
    "PERIOD",
    "APOAPSIS",
    "PERIAPSIS",
]

# Columns to drop before final export — metadata, constants, raw TLE lines,
# and orbital elements not useful for decay prediction
DROP_COLUMNS = [
    # Orbital params not predictive for decay
    "MEAN_ANOMALY",
    "REV_AT_EPOCH",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    # Constant / metadata across all records
    "CCSDS_OMM_VERS",
    "COMMENT",
    "ORIGINATOR",
    "CENTER_NAME",
    "REF_FRAME",
    "TIME_SYSTEM",
    "MEAN_ELEMENT_THEORY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "ELEMENT_SET_NO",
    # Raw TLE lines (already parsed into structured fields)
    "TLE_LINE0",
    "TLE_LINE1",
    "TLE_LINE2",
    # File / record identifiers
    "FILE",
    "GP_ID",
    # GP-native DECAY_DATE (nearly all null; label comes from CATALOG_DECAY_DATE)
    "DECAY_DATE",
]

# Rolling window sizes (in number of TLE records, not time)
ROLLING_WINDOWS = [5, 10, 20]


# ==================================================
# STEP 1: LOAD
# ==================================================
def load_data():
    print("=" * 60)
    print("STEP 1: Loading deduped dataset")
    print("=" * 60)

    df = pd.read_parquet(INPUT_PATH)
    print(f"  Rows:       {len(df):,}")
    print(f"  Satellites: {df['NORAD_CAT_ID'].nunique()}")
    print(f"  Columns:    {len(df.columns)}")
    return df


# ==================================================
# STEP 2: CAST NUMERIC COLUMNS
# ==================================================
def cast_numerics(df):
    print(f"\n{'=' * 60}")
    print("STEP 2: Casting numeric columns (string → float64)")
    print("=" * 60)

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            print(f"  ⚠ {col} not found — skipping")
            continue

        before_nulls = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_nulls = df[col].isna().sum() - before_nulls

        status = f"{new_nulls} failed" if new_nulls > 0 else "OK"
        print(f"  {col}: {status} → {df[col].dtype}")

    return df


# ==================================================
# STEP 3: COMPUTE LABEL — days_to_decay
# ==================================================
def compute_label(df):
    print(f"\n{'=' * 60}")
    print("STEP 3: Computing label (days_to_decay)")
    print("=" * 60)

    # Parse dates — keep as internal temporaries, don't overwrite raw strings
    epoch_dt = pd.to_datetime(df["EPOCH"], errors="coerce", utc=True)
    decay_dt = pd.to_datetime(df["CATALOG_DECAY_DATE"], errors="coerce", utc=True)

    print(f"  EPOCH parse failures:              {epoch_dt.isna().sum():,}")
    print(f"  CATALOG_DECAY_DATE parse failures: {decay_dt.isna().sum():,}")

    df["days_to_decay"] = (decay_dt - epoch_dt).dt.total_seconds() / SECONDS_PER_DAY

    valid = df["days_to_decay"].notna().sum()
    print(f"  Label valid: {valid:,} / {len(df):,}")
    print(f"  Range: [{df['days_to_decay'].min():.2f}, {df['days_to_decay'].max():.2f}] days")

    # Store parsed epoch for use in trend features (will drop later)
    df["_epoch_dt"] = epoch_dt

    return df


# ==================================================
# STEP 4: ROLLING / TREND FEATURES
# ==================================================
def compute_trend_features(df):
    print(f"\n{'=' * 60}")
    print("STEP 4: Computing per-satellite rolling/trend features")
    print("=" * 60)

    # Sort by satellite then time
    df = df.sort_values(["NORAD_CAT_ID", "_epoch_dt"]).reset_index(drop=True)

    # Time deltas between consecutive TLEs (in days), per satellite
    df["_dt_days"] = (
        df.groupby("NORAD_CAT_ID")["_epoch_dt"]
        .diff()
        .dt.total_seconds() / SECONDS_PER_DAY
    )

    # --- Rate-of-change features ---
    # For each key orbital parameter, compute Δvalue / Δtime

    rate_params = {
        "MEAN_MOTION":  "mean_motion_rate",    # rev/day per day — drag acceleration
        "BSTAR":        "bstar_rate",          # drag coefficient trend
        "ECCENTRICITY": "eccentricity_rate",   # orbit circularisation rate
        "PERIAPSIS":    "periapsis_rate",       # altitude decay rate (km/day)
    }

    for src_col, rate_col in rate_params.items():
        if src_col not in df.columns:
            print(f"  ⚠ {src_col} not found — skipping {rate_col}")
            continue

        delta = df.groupby("NORAD_CAT_ID")[src_col].diff()
        df[rate_col] = delta / df["_dt_days"]
        inf_count = np.isinf(df[rate_col]).sum()
        if inf_count > 0:
            df[rate_col] = df[rate_col].replace([np.inf, -np.inf], np.nan)
        print(f"  {rate_col}: computed ({df[rate_col].notna().sum():,} valid)")

    # --- Rolling mean features ---
    # Smooth out noise with rolling windows per satellite

    rolling_params = ["MEAN_MOTION", "BSTAR", "ECCENTRICITY", "PERIAPSIS"]

    for window in ROLLING_WINDOWS:
        for col in rolling_params:
            if col not in df.columns:
                continue

            roll_col = f"{col.lower()}_roll{window}"
            df[roll_col] = (
                df.groupby("NORAD_CAT_ID")[col]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
            )

        print(f"  Rolling window {window}: computed for {rolling_params}")

    # --- Rolling std for BSTAR (volatility indicator) ---
    for window in ROLLING_WINDOWS:
        std_col = f"bstar_std{window}"
        df[std_col] = (
            df.groupby("NORAD_CAT_ID")["BSTAR"]
            .transform(lambda s: s.rolling(window, min_periods=2).std())
        )

    print(f"  BSTAR rolling std: computed for windows {ROLLING_WINDOWS}")

    # --- Orbital lifetime position ---
    # Fraction of observed lifetime elapsed (0 = first TLE, 1 = last TLE)
    df["lifetime_fraction"] = (
        df.groupby("NORAD_CAT_ID")
        .cumcount()
        / df.groupby("NORAD_CAT_ID")["NORAD_CAT_ID"].transform("count").sub(1).clip(lower=1)
    )

    print(f"  lifetime_fraction: computed")

    # Clean up internal columns
    df = df.drop(columns=["_epoch_dt", "_dt_days"])

    return df


# ==================================================
# STEP 5: HANDLE NEGATIVE days_to_decay
# ==================================================
def handle_negative_decay(df):
    print(f"\n{'=' * 60}")
    print("STEP 5: Handling negative days_to_decay")
    print("=" * 60)

    neg = df["days_to_decay"] < 0
    n_neg = neg.sum()

    if n_neg == 0:
        print("  No negative values. Clean dataset.")
        return df

    small = neg & (df["days_to_decay"].abs() <= NEGATIVE_CLIP_THRESHOLD_DAYS)
    large = neg & (df["days_to_decay"].abs() > NEGATIVE_CLIP_THRESHOLD_DAYS)

    print(f"  Negative rows total:  {n_neg:,}")
    print(f"    ≤ {NEGATIVE_CLIP_THRESHOLD_DAYS} day (noise): {small.sum():,} → clipped to 0")
    print(f"    > {NEGATIVE_CLIP_THRESHOLD_DAYS} day (review): {large.sum():,}")

    df = df.copy()
    df.loc[small, "days_to_decay"] = 0.0

    if large.sum() > 0:
        print(f"\n  ⚠ Large negative rows (NOT auto-removed):")
        show = ["NORAD_CAT_ID", "EPOCH", "CATALOG_DECAY_DATE", "days_to_decay"]
        show = [c for c in show if c in df.columns]
        print(df.loc[large, show].to_string(index=False, max_rows=30))

    return df


# ==================================================
# STEP 6: DROP UNNECESSARY COLUMNS
# ==================================================
def drop_columns(df):
    print(f"\n{'=' * 60}")
    print("STEP 6: Dropping columns not useful for modeling")
    print("=" * 60)

    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    absent = [c for c in DROP_COLUMNS if c not in df.columns]

    df = df.drop(columns=to_drop)

    print(f"  Dropped {len(to_drop)} columns")
    if absent:
        print(f"  Already absent: {absent}")
    print(f"  Remaining columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"    {c}: {df[c].dtype}")

    return df


# ==================================================
# STEP 7: EXPORT
# ==================================================
def export(df):
    print(f"\n{'=' * 60}")
    print("STEP 7: Exporting final feature dataset")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    print(f"  Parquet: {OUTPUT_PARQUET}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV:     {OUTPUT_CSV}")


# ==================================================
# MAIN
# ==================================================
def main():
    df = load_data()
    df = cast_numerics(df)
    df = compute_label(df)
    df = compute_trend_features(df)
    df = handle_negative_decay(df)
    df = drop_columns(df)
    export(df)

    print(f"\n{'=' * 60}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"  Rows:          {len(df):,}")
    print(f"  Columns:       {len(df.columns)}")
    print(f"  Satellites:    {df['NORAD_CAT_ID'].nunique()}")
    label = df["days_to_decay"]
    print(f"  Label valid:   {label.notna().sum():,} / {len(df):,}")
    print(f"  Label range:   [{label.min():.2f}, {label.max():.2f}] days")
    print("=" * 60)


if __name__ == "__main__":
    main()
