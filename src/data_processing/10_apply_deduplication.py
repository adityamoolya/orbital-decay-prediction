"""
10_apply_deduplication.py

Applies the dedup rule confirmed by 09_repeated_epoch_analysis.py and produces
the final modeling dataset. Does NOT touch the raw/curated archive — reads it,
writes a separate modeling table.

Rule: within each (NORAD_CAT_ID, EPOCH) group, keep the row with the highest
GP_ID, tiebreak on latest CREATION_DATE. Confirmed safe by 09's chronology
check (99.98% agreement) and negligible orbital-parameter spread within groups.
"""

import pandas as pd

CURATED_PATH = "data/preprocessed/tle_features.parquet"   # the 950,677-row curated archive — read-only
OUTPUT_PATH = "data/preprocessed/tle_features_modeling.parquet"
OUTPUT_CSV = "data/preprocessed/tle_features_modeling.csv"

COL_NORAD = "NORAD_CAT_ID"
COL_EPOCH = "EPOCH"
COL_GP_ID = "GP_ID"
COL_CREATION_DATE = "CREATION_DATE"
COL_DECAY_DATE = "DECAY_DATE"              # GP-native, expected sparse
COL_CATALOG_DECAY_DATE = "CATALOG_DECAY_DATE"  # CelesTrak-joined, should be your actual label source
COL_DAYS_TO_DECAY = "days_to_decay"        # may not exist yet in this table — see check below

NEGATIVE_CLIP_THRESHOLD_DAYS = 1.0   # negative values smaller than this in magnitude = expected tracking-latency noise


def verify_label_source(df: pd.DataFrame):
    if COL_DECAY_DATE in df.columns:
        native_valid = df[COL_DECAY_DATE].notna().sum()
        print(f"  {COL_DECAY_DATE} (GP-native, expected sparse): {native_valid} valid / {len(df)} rows")

    if COL_CATALOG_DECAY_DATE in df.columns:
        catalog_valid = df[COL_CATALOG_DECAY_DATE].notna().sum()
        print(f"  {COL_CATALOG_DECAY_DATE} (catalog-joined, expected ~complete): {catalog_valid} valid / {len(df)} rows")
    else:
        print(f"  ⚠ {COL_CATALOG_DECAY_DATE} not found in this table at all.")

    if COL_DAYS_TO_DECAY not in df.columns:
        print(f"  ⚠ {COL_DAYS_TO_DECAY} column does NOT exist in this table.")
        print(f"    This means this is the pre-feature curated table, not build_features.py's output.")
        print(f"    Confirm you're pointing CURATED_PATH at the right file before continuing.")
        return False

    label_valid = df[COL_DAYS_TO_DECAY].notna().sum()
    print(f"  {COL_DAYS_TO_DECAY} (final label): {label_valid} valid / {len(df)} rows")

    if COL_CATALOG_DECAY_DATE in df.columns:
        if label_valid < 0.9 * df[COL_CATALOG_DECAY_DATE].notna().sum():
            print("  ⚠ WARNING: days_to_decay has far fewer valid rows than CATALOG_DECAY_DATE — "
                  "check whether the label was built from the wrong decay-date field.")
        else:
            print("  ✓ Label coverage matches catalog decay-date coverage.")
    return True

def apply_dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    sort_cols = [COL_NORAD, COL_EPOCH]
    sort_order = [True, True]
    if COL_GP_ID in df.columns:
        sort_cols += [COL_GP_ID]
        sort_order += [False]  # highest GP_ID first
    if COL_CREATION_DATE in df.columns:
        sort_cols += [COL_CREATION_DATE]
        sort_order += [False]  # latest CREATION_DATE as tiebreak

    df_sorted = df.sort_values(by=sort_cols, ascending=sort_order)
    df_deduped = df_sorted.drop_duplicates(subset=[COL_NORAD, COL_EPOCH], keep="first")

    after = len(df_deduped)
    print(f"  Rows before dedup: {before:,}")
    print(f"  Rows after dedup:  {after:,}")
    print(f"  Removed:           {before - after:,}")

    return df_deduped.sort_values(by=[COL_NORAD, COL_EPOCH]).reset_index(drop=True)


def handle_negative_decay(df: pd.DataFrame) -> pd.DataFrame:
    negative_mask = df[COL_DAYS_TO_DECAY] < 0
    n_negative = negative_mask.sum()

    if n_negative == 0:
        print("  No negative days_to_decay rows found.")
        return df

    small_mag = negative_mask & (df[COL_DAYS_TO_DECAY].abs() <= NEGATIVE_CLIP_THRESHOLD_DAYS)
    large_mag = negative_mask & (df[COL_DAYS_TO_DECAY].abs() > NEGATIVE_CLIP_THRESHOLD_DAYS)

    print(f"  Negative days_to_decay rows: {n_negative}")
    print(f"    Small magnitude (<= {NEGATIVE_CLIP_THRESHOLD_DAYS} day, expected tracking-latency noise): {small_mag.sum()}")
    print(f"    Large magnitude (> {NEGATIVE_CLIP_THRESHOLD_DAYS} day, NEEDS MANUAL REVIEW): {large_mag.sum()}")

    if large_mag.sum() > 0:
        print("  ⚠ Large-magnitude negative rows found — NOT auto-clipped. Inspect these before proceeding:")
        print(df.loc[large_mag, [COL_NORAD, COL_EPOCH, COL_DAYS_TO_DECAY]].to_string(index=False))

    df = df.copy()
    df.loc[small_mag, COL_DAYS_TO_DECAY] = 0.0

    return df


def main():
    print("Loading curated dataset...")
    df = pd.read_parquet(CURATED_PATH)
    print(f"Loaded {len(df):,} rows, {df[COL_NORAD].nunique()} satellites\n")

    print("========== VERIFYING LABEL SOURCE ==========")
    has_label = verify_label_source(df)
    if not has_label:
        print("\nStopping — fix the label/path issue above before deduplicating.")
        return

    print("========== VERIFYING LABEL SOURCE ==========")
    verify_label_source(df)

    print("\n========== APPLYING DEDUPLICATION ==========")
    df = apply_dedup(df)

    print("\n========== HANDLING NEGATIVE DAYS_TO_DECAY ==========")
    df = handle_negative_decay(df)

    print(f"\n========== FINAL MODELING DATASET ==========")
    print(f"  Rows: {len(df):,}")
    print(f"  Satellites: {df[COL_NORAD].nunique()}")

    df.to_parquet(OUTPUT_PATH, index=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWritten to:\n  {OUTPUT_PATH}\n  {OUTPUT_CSV}")
    print(f"\nOriginal curated archive at {CURATED_PATH} is untouched.")


if __name__ == "__main__":
    main()