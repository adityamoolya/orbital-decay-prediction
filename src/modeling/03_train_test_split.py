"""
03_train_test_split.py

Performs a strict satellite-level train/test split (80/20) for decay prediction:
  1. Identifies all unique satellites (NORAD_CAT_ID).
  2. Splits satellites at the entity level using a fixed seed (80% train, 20% test).
     Zero satellite overlap: every observation of a given satellite belongs
     strictly to train OR strictly to test.
  3. Saves locked satellite ID lists to data/splits/test_satellites.csv and
     data/splits/train_satellites.csv to guarantee deterministic reproducibility.
  4. Handles initial rolling/difference boundary NaNs (195 rows, exactly 1 per satellite)
     with clear explanation and logging.
  5. Computes and compares distribution statistics between train and test splits:
       - Satellite lifespan (TLE observations per satellite: mean, median, min, max)
       - Altitude proxy (mean PERIAPSIS and SEMIMAJOR_AXIS)
       - Target distribution (mean, median, min, max days_to_decay)
  6. Evaluates balance to sanity-check against skew (suggests re-seeding if skewed).
  7. Exports split datasets:
       data/modeling/train_features.parquet
       data/modeling/test_features.parquet

Input:  data/modeling/features.parquet (832,434 rows, 195 satellites)
Output: data/splits/train_satellites.csv
        data/splits/test_satellites.csv
        data/modeling/train_features.parquet
        data/modeling/test_features.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data" / "modeling" / "features.parquet"

SPLITS_DIR = BASE_DIR / "data" / "splits"
TRAIN_SATS_CSV = SPLITS_DIR / "train_satellites.csv"
TEST_SATS_CSV = SPLITS_DIR / "test_satellites.csv"

OUTPUT_TRAIN_PARQUET = BASE_DIR / "data" / "modeling" / "train_features.parquet"
OUTPUT_TEST_PARQUET = BASE_DIR / "data" / "modeling" / "test_features.parquet"

RANDOM_SEED = 42
TEST_SIZE = 0.20  # 80/20 satellite-level split

# Strategy for handling boundary NaNs (1st observation per satellite has no prior TLE)
# Option 'drop': Drops the 195 initial boundary rows (0.023% of data) — mathematically cleanest.
# Option 'impute_bfill': Backward fills within each satellite so initial rates match the 2nd TLE.
NAN_STRATEGY = "drop"


# ==================================================
# 1. LOAD DATASET
# ==================================================
def load_data():
    print("=" * 70)
    print("1. LOADING FEATURE MATRIX")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found at: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    print(f"  Loaded rows:       {len(df):,}")
    print(f"  Total satellites:  {df['NORAD_CAT_ID'].nunique()}")
    print(f"  Columns:           {len(df.columns)}")

    return df


# ==================================================
# 2. HANDLE BOUNDARY NaNs FROM ROLLING / DIFF
# ==================================================
def handle_boundary_nans(df, strategy=NAN_STRATEGY):
    print(f"\n{'=' * 70}")
    print(f"2. HANDLING INITIAL BOUNDARY NaNs (Strategy: '{strategy}')")
    print("=" * 70)

    # Rate and rolling std columns contain exactly 195 NaNs (1 per satellite at t_0)
    diff_cols = [
        "mean_motion_rate", "bstar_rate", "eccentricity_rate", "periapsis_rate",
        "bstar_std5", "bstar_std10", "bstar_std20"
    ]
    diff_cols = [c for c in diff_cols if c in df.columns]

    initial_nulls = df[diff_cols].isna().sum().max()
    print(f"  Identified {initial_nulls} boundary NaN rows across dynamic rate/std features.")
    print("  Physical cause: At the initial observation (t_0) of each satellite,")
    print("  numerical time differencing (Δparam / Δt) is undefined because no")
    print("  preceding TLE observation exists.")

    if strategy == "drop":
        print("\n  Decision: DROPPING the 195 boundary rows.")
        print("  Rationale:")
        print(f"    - Represents only {initial_nulls / len(df) * 100:.4f}% of the dataset (195 / {len(df):,}).")
        print("    - Avoids injecting synthetic zero-derivatives or artificial rate assumptions.")
        print("    - Ensures all numerical features fed to models are 100% genuine and complete.")

        rows_before = len(df)
        df_clean = df.dropna(subset=diff_cols).copy()
        print(f"  Rows retained: {len(df_clean):,} (dropped {rows_before - len(df_clean):,} rows)")
        return df_clean

    elif strategy == "impute_bfill":
        print("\n  Decision: IMPUTING via backward-fill within each satellite.")
        print("  Rationale: Preserves 100% of rows by assuming the rate of change at t_0")
        print("  approximates the subsequent observed velocity at t_1.")

        df_clean = df.copy()
        for col in diff_cols:
            df_clean[col] = (
                df_clean.groupby("NORAD_CAT_ID")[col]
                .transform(lambda s: s.bfill().fillna(0.0))
            )
        print(f"  Imputed {initial_nulls} values across {len(diff_cols)} columns.")
        return df_clean

    else:
        raise ValueError(f"Unknown NaN strategy: {strategy}")


# ==================================================
# 3. SATELLITE-LEVEL SPLIT (LOCKED VIA CSV)
# ==================================================
def perform_satellite_split(df):
    print(f"\n{'=' * 70}")
    print("3. SATELLITE-LEVEL SPLITTING (80% TRAIN / 20% TEST)")
    print("=" * 70)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    all_satellites = np.sort(df["NORAD_CAT_ID"].unique())
    total_sats = len(all_satellites)

    # Check if locked split already exists
    if TRAIN_SATS_CSV.exists() and TEST_SATS_CSV.exists():
        print("  ✓ Found existing locked split files in data/splits/ — loading them:")
        train_sats = pd.read_csv(TRAIN_SATS_CSV)["NORAD_CAT_ID"].values
        test_sats = pd.read_csv(TEST_SATS_CSV)["NORAD_CAT_ID"].values
        print(f"    Loaded {len(train_sats)} train satellites from {TRAIN_SATS_CSV.name}")
        print(f"    Loaded {len(test_sats)} test satellites from {TEST_SATS_CSV.name}")
    else:
        print(f"  Generating reproducible split with seed={RANDOM_SEED}...")
        train_sats, test_sats = train_test_split(
            all_satellites,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )

        train_sats = np.sort(train_sats)
        test_sats = np.sort(test_sats)

        pd.DataFrame({"NORAD_CAT_ID": train_sats}).to_csv(TRAIN_SATS_CSV, index=False)
        pd.DataFrame({"NORAD_CAT_ID": test_sats}).to_csv(TEST_SATS_CSV, index=False)

        print(f"  ✓ Locked train satellites saved to: {TRAIN_SATS_CSV}")
        print(f"  ✓ Locked test satellites saved to:  {TEST_SATS_CSV}")

    # Verify zero overlap
    overlap = set(train_sats).intersection(set(test_sats))
    assert len(overlap) == 0, f"Critical error: {len(overlap)} satellites overlap between train and test!"
    print(f"  ✓ Zero overlap verified: 0 satellites appear in both splits.")

    train_df = df[df["NORAD_CAT_ID"].isin(train_sats)].copy()
    test_df = df[df["NORAD_CAT_ID"].isin(test_sats)].copy()

    print(f"\n  Satellite Counts: Total = {total_sats} | Train = {len(train_sats)} ({len(train_sats)/total_sats*100:.1f}%) | Test = {len(test_sats)} ({len(test_sats)/total_sats*100:.1f}%)")
    print(f"  Observation Rows: Total = {len(df):,} | Train = {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%) | Test = {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, test_df, train_sats, test_sats


# ==================================================
# 4. BALANCE & SKEW SANITY CHECK
# ==================================================
def compare_splits(train_df, test_df):
    print(f"\n{'=' * 70}")
    print("4. SPLIT BALANCE & SKEW SANITY CHECK")
    print("=" * 70)

    # 1. Lifespan (TLE observations per satellite)
    train_counts = train_df.groupby("NORAD_CAT_ID").size()
    test_counts = test_df.groupby("NORAD_CAT_ID").size()

    # 2. Altitude proxy (Mean Periapsis per satellite in km)
    train_perigee = train_df.groupby("NORAD_CAT_ID")["PERIAPSIS"].mean()
    test_perigee = test_df.groupby("NORAD_CAT_ID")["PERIAPSIS"].mean()

    # 3. Semi-major axis per satellite (km)
    train_sma = train_df.groupby("NORAD_CAT_ID")["SEMIMAJOR_AXIS"].mean()
    test_sma = test_df.groupby("NORAD_CAT_ID")["SEMIMAJOR_AXIS"].mean()

    # 4. Target variable (days_to_decay)
    train_target = train_df["days_to_decay"]
    test_target = test_df["days_to_decay"]

    stats_table = [
        {
            "Metric": "Satellite Count",
            "Train Split": f"{len(train_counts):,}",
            "Test Split": f"{len(test_counts):,}",
            "Ratio / Diff": f"{len(test_counts)/(len(train_counts)+len(test_counts))*100:.1f}% test",
        },
        {
            "Metric": "Observation Rows",
            "Train Split": f"{len(train_df):,}",
            "Test Split": f"{len(test_df):,}",
            "Ratio / Diff": f"{len(test_df)/(len(train_df)+len(test_df))*100:.1f}% test",
        },
        {
            "Metric": "TLEs per Sat (Mean)",
            "Train Split": f"{train_counts.mean():.1f}",
            "Test Split": f"{test_counts.mean():.1f}",
            "Ratio / Diff": f"{abs(train_counts.mean() - test_counts.mean()) / train_counts.mean() * 100:.1f}% diff",
        },
        {
            "Metric": "TLEs per Sat (Median)",
            "Train Split": f"{train_counts.median():.0f}",
            "Test Split": f"{test_counts.median():.0f}",
            "Ratio / Diff": f"{abs(train_counts.median() - test_counts.median()):.0f} diff",
        },
        {
            "Metric": "TLEs per Sat (Min - Max)",
            "Train Split": f"{train_counts.min():,} - {train_counts.max():,}",
            "Test Split": f"{test_counts.min():,} - {test_counts.max():,}",
            "Ratio / Diff": "Comparable",
        },
        {
            "Metric": "Mean Periapsis (km)",
            "Train Split": f"{train_perigee.mean():.2f}",
            "Test Split": f"{test_perigee.mean():.2f}",
            "Ratio / Diff": f"{abs(train_perigee.mean() - test_perigee.mean()):.2f} km",
        },
        {
            "Metric": "Median Periapsis (km)",
            "Train Split": f"{train_perigee.median():.2f}",
            "Test Split": f"{test_perigee.median():.2f}",
            "Ratio / Diff": f"{abs(train_perigee.median() - test_perigee.median()):.2f} km",
        },
        {
            "Metric": "Mean Semi-Major Axis (km)",
            "Train Split": f"{train_sma.mean():.2f}",
            "Test Split": f"{test_sma.mean():.2f}",
            "Ratio / Diff": f"{abs(train_sma.mean() - test_sma.mean()):.2f} km",
        },
        {
            "Metric": "Mean days_to_decay (days)",
            "Train Split": f"{train_target.mean():.2f}",
            "Test Split": f"{test_target.mean():.2f}",
            "Ratio / Diff": f"{abs(train_target.mean() - test_target.mean()):.2f} days",
        },
        {
            "Metric": "Median days_to_decay (days)",
            "Train Split": f"{train_target.median():.2f}",
            "Test Split": f"{test_target.median():.2f}",
            "Ratio / Diff": f"{abs(train_target.median() - test_target.median()):.2f} days",
        },
    ]

    report_df = pd.DataFrame(stats_table)
    print(report_df.to_string(index=False))

    # Skew evaluation rule of thumb
    periapsis_diff_pct = abs(train_perigee.mean() - test_perigee.mean()) / train_perigee.mean() * 100
    lifespan_diff_pct = abs(train_counts.median() - test_counts.median()) / train_counts.median() * 100

    print("\n  Sanity Check Assessment:")
    if periapsis_diff_pct < 10.0 and lifespan_diff_pct < 35.0:
        print("  ✓ Split is WELL-BALANCED across altitude and satellite lifespan.")
        print("  ✓ No pathological skew detected (test set has representative long and short-lived orbits).")
    else:
        print("  ⚠ Note: Noticeable divergence between train and test distribution.")
        print(f"    - Periapsis altitude diff: {periapsis_diff_pct:.1f}%")
        print(f"    - Lifespan median diff:   {lifespan_diff_pct:.1f}%")
        print("    Recommendation: Re-seed RANDOM_SEED (e.g. 7, 21, 100) if a tighter balance is preferred.")


# ==================================================
# 5. EXPORT SPLIT DATASETS
# ==================================================
def export_splits(train_df, test_df):
    print(f"\n{'=' * 70}")
    print("5. EXPORTING TRAIN & TEST DATASETS")
    print("=" * 70)

    train_df.to_parquet(OUTPUT_TRAIN_PARQUET, index=False, engine="pyarrow")
    test_df.to_parquet(OUTPUT_TEST_PARQUET, index=False, engine="pyarrow")

    print(f"  ✓ Train dataset saved: {OUTPUT_TRAIN_PARQUET}")
    print(f"    Rows: {len(train_df):,} | Satellites: {train_df['NORAD_CAT_ID'].nunique()}")
    print(f"  ✓ Test dataset saved:  {OUTPUT_TEST_PARQUET}")
    print(f"    Rows: {len(test_df):,} | Satellites: {test_df['NORAD_CAT_ID'].nunique()}")


# ==================================================
# MAIN PIPELINE
# ==================================================
def main():
    df = load_data()
    df_clean = handle_boundary_nans(df)
    train_df, test_df, train_sats, test_sats = perform_satellite_split(df_clean)
    compare_splits(train_df, test_df)
    export_splits(train_df, test_df)

    print(f"\n{'=' * 70}")
    print("SATELLITE-LEVEL SPLIT COMPLETED SUCCESSFULLY")
    print(f"  Seed:       {RANDOM_SEED}")
    print(f"  Train Sats: {len(train_sats)} ({len(train_df):,} rows)")
    print(f"  Test Sats:  {len(test_sats)} ({len(test_df):,} rows)")
    print(f"  Lockfiles:  {TRAIN_SATS_CSV} and {TEST_SATS_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
