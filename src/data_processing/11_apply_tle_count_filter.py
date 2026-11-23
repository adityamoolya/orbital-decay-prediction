"""
11_apply_tle_count_filter.py

Addresses the two issues identified after deduplication:

  1. The deduped table (tle_features_modeling.parquet) still contains 204
     satellites — including 9 satellites with fewer than 10 TLE records
     (debris fragments flagged by verify_tle_jsons.py). These must be
     excluded before modeling.

  2. The days_to_decay label does not exist yet in this table (dedup ran
     before feature extraction). This script does NOT compute it — that
     is the job of build_features.py. This script simply applies the
     TLE-count exclusion filter and saves a clean table ready for
     feature extraction.

Input:  data/preprocessed/tle_features_modeling.parquet  (832,452 rows, 204 sats)
Output: data/preprocessed/tle_features_modeling.parquet  (overwritten, ~195 sats)
        data/preprocessed/tle_features_modeling.csv      (overwritten)
"""

import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
INPUT_PATH  = "data/preprocessed/tle_features_modeling.parquet"
OUTPUT_PATH = "data/preprocessed/tle_features_modeling.parquet"
OUTPUT_CSV  = "data/preprocessed/tle_features_modeling.csv"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
MIN_TLE_COUNT = 10


def main():
    print("=" * 60)
    print("LOADING DEDUPED DATASET")
    print("=" * 60)

    df = pd.read_parquet(INPUT_PATH)
    print(f"  Rows:       {len(df):,}")
    print(f"  Satellites: {df['NORAD_CAT_ID'].nunique()}")
    print(f"  Columns:    {len(df.columns)}")

    # ----- TLE count distribution -----
    counts = df.groupby("NORAD_CAT_ID").size()

    print(f"\n{'=' * 60}")
    print(f"FILTERING SATELLITES WITH < {MIN_TLE_COUNT} TLE RECORDS")
    print("=" * 60)

    low = counts[counts < MIN_TLE_COUNT]

    if len(low) == 0:
        print(f"  All satellites have >= {MIN_TLE_COUNT} records. Nothing to remove.")
    else:
        print(f"  Found {len(low)} satellites with < {MIN_TLE_COUNT} records:")
        for norad_id, cnt in sorted(low.items()):
            print(f"    NORAD {norad_id}: {cnt} record(s)")

        rows_before = len(df)
        df = df[~df["NORAD_CAT_ID"].isin(low.index)].copy()
        rows_removed = rows_before - len(df)

        print(f"\n  Removed {len(low)} satellites ({rows_removed:,} rows)")

    # ----- Confirm label status -----
    print(f"\n{'=' * 60}")
    print("LABEL STATUS CHECK")
    print("=" * 60)

    if "days_to_decay" in df.columns:
        print("  days_to_decay: PRESENT")
    else:
        print("  days_to_decay: NOT PRESENT (expected — will be computed by build_features.py)")

    if "CATALOG_DECAY_DATE" in df.columns:
        valid = df["CATALOG_DECAY_DATE"].notna().sum()
        print(f"  CATALOG_DECAY_DATE: {valid:,} / {len(df):,} valid (label source for build_features.py)")
    else:
        print("  ⚠ CATALOG_DECAY_DATE: MISSING — check pipeline")

    if "DECAY_DATE" in df.columns:
        valid = df["DECAY_DATE"].notna().sum()
        print(f"  DECAY_DATE (GP-native, sparse): {valid:,} / {len(df):,} valid")

    # ----- Save -----
    print(f"\n{'=' * 60}")
    print("SAVING FILTERED DATASET")
    print("=" * 60)

    df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"  Parquet: {OUTPUT_PATH}")
    print(f"  CSV:     {OUTPUT_CSV}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"  Final rows:       {len(df):,}")
    print(f"  Final satellites:  {df['NORAD_CAT_ID'].nunique()}")
    print(f"  Next step:        Run build_features.py against this table")
    print("=" * 60)


if __name__ == "__main__":
    main()
