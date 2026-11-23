
"""
Investigates duplicate records in the curated satellite dataset.

Checks:
1. Exact duplicate rows across all columns.
2. Repeated NORAD_CAT_ID + EPOCH combinations.
3. Whether repeated combinations have conflicting values.
4. Differences in source metadata such as GP_ID, FILE and TLE lines.
5. Satellites with the most repeated epoch records.

The curated dataset is read-only. No records are removed.
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/curated/tle_features.parquet")
OUTPUT_DIR = Path("reports/eda/duplicates")

KEY_COLUMNS = ["NORAD_CAT_ID", "EPOCH"]

# Limit the detailed sample of exact duplicate rows.
EXACT_DUPLICATE_SAMPLE_SIZE = 5000


def load_data():
    print("Loading curated dataset...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Records loaded: {len(df):,}")
    return df


def investigate_exact_duplicates(df):
    print("\n========== EXACT DUPLICATE ROWS ==========")

    # A row is an exact duplicate only if every column matches.
    duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = df.loc[duplicate_mask].copy()

    duplicate_excess = int(df.duplicated(keep="first").sum())
    duplicate_groups = int(
        df.loc[df.duplicated(keep=False)]
        .drop_duplicates()
        .shape[0]
    )

    print(f"Rows belonging to exact duplicate groups: {len(duplicate_rows):,}")
    print(f"Excess exact duplicate rows: {duplicate_excess:,}")
    print(f"Unique rows represented by duplicate groups: {duplicate_groups:,}")

    if not duplicate_rows.empty:
        sample = duplicate_rows.head(EXACT_DUPLICATE_SAMPLE_SIZE)
        sample.to_csv(
            OUTPUT_DIR / "exact_duplicate_sample.csv",
            index=False,
        )

    return {
        "rows_in_exact_duplicate_groups": len(duplicate_rows),
        "excess_exact_duplicate_rows": duplicate_excess,
        "unique_rows_in_duplicate_groups": duplicate_groups,
    }


def investigate_repeated_epochs(df):
    print("\n========== REPEATED SATELLITE EPOCHS ==========")

    # Parse epochs separately. Keep original EPOCH unchanged.
    work = df.copy()
    work["_EPOCH_PARSED"] = pd.to_datetime(
        work["EPOCH"], errors="coerce", utc=True
    )

    invalid_epochs = int(work["_EPOCH_PARSED"].isna().sum())
    print(f"Invalid/unparseable epochs: {invalid_epochs:,}")

    valid = work.dropna(
        subset=["NORAD_CAT_ID", "_EPOCH_PARSED"]
    ).copy()

    grouped = valid.groupby(
        ["NORAD_CAT_ID", "_EPOCH_PARSED"],
        dropna=False,
        sort=False,
    )

    sizes = grouped.size().rename("RECORD_COUNT")
    repeated = sizes[sizes > 1].reset_index()

    repeated["EXTRA_RECORDS"] = repeated["RECORD_COUNT"] - 1

    print(f"Unique satellite-epoch combinations: {len(sizes):,}")
    print(f"Repeated satellite-epoch combinations: {len(repeated):,}")
    print(
        "Rows belonging to repeated combinations: "
        f"{repeated['RECORD_COUNT'].sum():,}"
    )
    print(
        "Excess records across repeated combinations: "
        f"{repeated['EXTRA_RECORDS'].sum():,}"
    )

    repeated.to_csv(
        OUTPUT_DIR / "repeated_epoch_groups.csv",
        index=False,
    )

    return valid, repeated


def investigate_conflicts(valid, repeated):
    print("\n========== CONFLICTING REPEATED RECORDS ==========")

    if repeated.empty:
        print("No repeated satellite-epoch combinations found.")
        return pd.DataFrame()

    keys = ["NORAD_CAT_ID", "_EPOCH_PARSED"]

    # Compare every non-key column within each repeated group.
    compare_columns = [
        col for col in valid.columns
        if col not in keys and col != "EPOCH"
    ]

    repeated_keys = repeated[keys]

    repeated_records = valid.merge(
        repeated_keys,
        on=keys,
        how="inner",
    )

    grouped = repeated_records.groupby(
        keys,
        dropna=False,
        sort=False,
    )

    # Number of distinct values per column within each repeated group.
    nunique = grouped[compare_columns].nunique(dropna=False)

    conflict_mask = nunique.gt(1).any(axis=1)
    conflicting = nunique.loc[conflict_mask].copy()

    conflicting["CONFLICTING_COLUMNS"] = [
        ", ".join(row.index[row > 1].tolist())
        for _, row in conflicting.iterrows()
    ]
    conflicting["CONFLICT_COLUMN_COUNT"] = nunique.gt(1).sum(axis=1)

    conflicting = conflicting.reset_index()

    print(f"Repeated groups with conflicting values: {len(conflicting):,}")
    print(
        "Repeated groups with identical values across all compared "
        f"columns: {len(repeated) - len(conflicting):,}"
    )

    conflicting.to_csv(
        OUTPUT_DIR / "conflicting_epoch_groups.csv",
        index=False,
    )

    # Save all records belonging to conflicting groups for inspection.
    if not conflicting.empty:
        conflict_records = repeated_records.merge(
            conflicting[keys],
            on=keys,
            how="inner",
        )

        conflict_records.to_csv(
            OUTPUT_DIR / "conflicting_epoch_records.csv",
            index=False,
        )

    return conflicting


def satellite_duplicate_summary(repeated):
    print("\n========== SATELLITE DUPLICATE SUMMARY ==========")

    if repeated.empty:
        return

    summary = (
        repeated.groupby("NORAD_CAT_ID")
        .agg(
            REPEATED_EPOCH_GROUPS=("RECORD_COUNT", "size"),
            ROWS_IN_REPEATED_GROUPS=("RECORD_COUNT", "sum"),
            EXTRA_RECORDS=("EXTRA_RECORDS", "sum"),
            MAX_RECORDS_AT_ONE_EPOCH=("RECORD_COUNT", "max"),
        )
        .sort_values("EXTRA_RECORDS", ascending=False)
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "satellite_duplicate_summary.csv",
        index=False,
    )

    print("\nTop 20 satellites by excess repeated records:")
    print(summary.head(20).to_string(index=False))


def main():
    print("Starting duplicate record investigation...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    missing_keys = [
        col for col in KEY_COLUMNS if col not in df.columns
    ]

    if missing_keys:
        raise ValueError(
            f"Required columns missing from dataset: {missing_keys}"
        )

    exact_summary = investigate_exact_duplicates(df)

    valid, repeated = investigate_repeated_epochs(df)

    conflicting = investigate_conflicts(valid, repeated)

    satellite_duplicate_summary(repeated)

    summary = pd.DataFrame([
        {
            "METRIC": key,
            "VALUE": value,
        }
        for key, value in exact_summary.items()
    ])

    summary = pd.concat(
        [
            summary,
            pd.DataFrame([
                {
                    "METRIC": "repeated_satellite_epoch_groups",
                    "VALUE": len(repeated),
                },
                {
                    "METRIC": "rows_in_repeated_epoch_groups",
                    "VALUE": int(repeated["RECORD_COUNT"].sum()),
                },
                {
                    "METRIC": "excess_repeated_epoch_records",
                    "VALUE": int(repeated["EXTRA_RECORDS"].sum()),
                },
                {
                    "METRIC": "repeated_groups_with_conflicts",
                    "VALUE": len(conflicting),
                },
            ]),
        ],
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT_DIR / "duplicate_investigation_summary.csv",
        index=False,
    )

    print("\n========== INVESTIGATION COMPLETED ==========")
    print(f"Reports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
