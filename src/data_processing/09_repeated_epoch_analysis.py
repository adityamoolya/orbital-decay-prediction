
"""
Repeated Epoch Analysis

Investigates repeated NORAD_CAT_ID + EPOCH combinations.

Checks:
1. GP_ID and CREATION_DATE ordering agreement.
2. Magnitude of orbital parameter differences.
3. High-repeat-count satellites and decay proximity.
4. Detailed records for NORAD 45755.
5. Source metadata differences.

No records are deleted or modified.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


INPUT_FILE = Path("data/curated/tle_features.parquet")
OUTPUT_DIR = Path("reports/eda/duplicates/repeated_epochs")

KEYS = ["NORAD_CAT_ID", "_EPOCH"]

ORBITAL_COLUMNS = [
    "MEAN_MOTION",
    "ECCENTRICITY",
    "BSTAR",
]

METADATA_COLUMNS = [
    "GP_ID",
    "FILE",
    "CREATION_DATE",
    "TLE_LINE1",
    "TLE_LINE2",
    "OBJECT_NAME",
    "OBJECT_TYPE",
]

DECAY_COLUMNS = [
    "DECAY_DATE",
    "CATALOG_DECAY_DATE",
]

HIGH_REPEAT_THRESHOLD = 3
FOCUS_NORAD_ID = 45755


def load_data():
    print("Loading curated dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Records loaded: {len(df):,}")

    required = ["NORAD_CAT_ID", "EPOCH"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["_EPOCH"] = pd.to_datetime(
        df["EPOCH"],
        errors="coerce",
        utc=True,
    )

    df["_CREATION_DATE"] = pd.to_datetime(
        df["CREATION_DATE"],
        errors="coerce",
        utc=True,
    )

    df["_GP_ID_NUMERIC"] = pd.to_numeric(
        df["GP_ID"],
        errors="coerce",
    )

    for col in ORBITAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    print(f"Invalid epochs: {df['_EPOCH'].isna().sum():,}")
    print(
        "Invalid GP_ID conversions: "
        f"{df['_GP_ID_NUMERIC'].isna().sum():,}"
    )

    return df


def get_repeated_records(df):
    print("\n========== IDENTIFYING REPEATED EPOCHS ==========")

    valid = df.dropna(
        subset=["NORAD_CAT_ID", "_EPOCH"]
    ).copy()

    sizes = (
        valid.groupby(KEYS, sort=False)
        .size()
        .rename("RECORD_COUNT")
        .reset_index()
    )

    repeated = sizes[sizes["RECORD_COUNT"] > 1].copy()

    records = valid.merge(
        repeated,
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )

    print(f"Repeated epoch groups: {len(repeated):,}")
    print(f"Records in repeated groups: {len(records):,}")
    print(
        "Excess records: "
        f"{records.shape[0] - len(repeated):,}"
    )

    repeated.to_csv(
        OUTPUT_DIR / "repeated_epoch_groups.csv",
        index=False,
    )

    return records, repeated


def investigate_chronology(records):
    print("\n========== GP_ID vs CREATION_DATE CHRONOLOGY ==========")

    results = []

    for (norad_id, epoch), group in records.groupby(
        KEYS,
        sort=False,
    ):
        usable = group.dropna(
            subset=["_GP_ID_NUMERIC", "_CREATION_DATE"]
        )

        discordant = 0
        comparable = 0
        tied_gp = 0
        tied_creation = 0

        rows = list(
            usable[
                ["_GP_ID_NUMERIC", "_CREATION_DATE"]
            ].itertuples(index=False, name=None)
        )

        for (gp1, date1), (gp2, date2) in combinations(rows, 2):

            if gp1 == gp2:
                tied_gp += 1
                continue

            if date1 == date2:
                tied_creation += 1
                continue

            comparable += 1

            if (gp1 - gp2) * (date1 - date2).total_seconds() < 0:
                discordant += 1

        if comparable == 0:
            status = "INSUFFICIENT_ORDERING_DATA"
        elif discordant == 0:
            status = "ORDERING_AGREES"
        else:
            status = "ORDERING_DISAGREES"

        results.append({
            "NORAD_CAT_ID": norad_id,
            "EPOCH": epoch,
            "RECORD_COUNT": len(group),
            "VALID_CHRONOLOGY_RECORDS": len(usable),
            "COMPARABLE_PAIRS": comparable,
            "DISCORDANT_PAIRS": discordant,
            "TIED_GP_ID_PAIRS": tied_gp,
            "TIED_CREATION_DATE_PAIRS": tied_creation,
            "DISCORDANCE_PERCENTAGE": (
                discordant / comparable * 100
                if comparable else np.nan
            ),
            "ORDERING_STATUS": status,
        })

    result = pd.DataFrame(results)

    result.to_csv(
        OUTPUT_DIR / "gp_id_creation_date_ordering.csv",
        index=False,
    )

    print("\nOrdering summary:")
    print(
        result["ORDERING_STATUS"]
        .value_counts(dropna=False)
        .to_string()
    )

    comparable = result["COMPARABLE_PAIRS"].sum()
    discordant = result["DISCORDANT_PAIRS"].sum()

    print(f"\nComparable pairs: {comparable:,}")
    print(f"Discordant pairs: {discordant:,}")

    if comparable:
        print(
            "Overall discordance: "
            f"{discordant / comparable * 100:.4f}%"
        )

    return result


def investigate_orbital_differences(records):
    print("\n========== ORBITAL SOLUTION DIFFERENCES ==========")

    available = [
        col for col in ORBITAL_COLUMNS
        if col in records.columns
    ]

    grouped = records.groupby(KEYS, sort=False)

    result = grouped.size().rename("RECORD_COUNT").to_frame()

    for col in available:
        stats = grouped[col].agg(
            ["min", "max", "mean", "std"]
        )

        result[f"{col}_MIN"] = stats["min"]
        result[f"{col}_MAX"] = stats["max"]
        result[f"{col}_DELTA"] = stats["max"] - stats["min"]
        result[f"{col}_STD"] = stats["std"]

    result = result.reset_index()

    result.to_csv(
        OUTPUT_DIR / "orbital_parameter_differences.csv",
        index=False,
    )

    print("\nMaximum observed differences across groups:")

    for col in available:
        delta = result[f"{col}_DELTA"]

        print(f"\n{col}")
        print(f"Median delta: {delta.median():.12g}")
        print(f"Mean delta:   {delta.mean():.12g}")
        print(f"Maximum delta: {delta.max():.12g}")
        print(f"Zero delta groups: {(delta == 0).sum():,}")
        print(f"Nonzero delta groups: {(delta > 0).sum():,}")

    return result


def investigate_metadata(records):
    print("\n========== SOURCE METADATA DIFFERENCES ==========")

    available = [
        col for col in METADATA_COLUMNS
        if col in records.columns
    ]

    grouped = records.groupby(KEYS, sort=False)

    result = grouped[available].nunique(
        dropna=False
    )

    result.columns = [
        f"{col}_UNIQUE_VALUES"
        for col in result.columns
    ]

    result = result.reset_index()

    unique_columns = [
        col for col in result.columns
        if col.endswith("_UNIQUE_VALUES")
    ]

    result["DIFFERING_METADATA_COLUMNS"] = result.apply(
        lambda row: ", ".join(
            col.replace("_UNIQUE_VALUES", "")
            for col in unique_columns
            if row[col] > 1
        ),
        axis=1,
    )

    result["DIFFERING_METADATA_COUNT"] = (
        result[unique_columns] > 1
    ).sum(axis=1)

    result.to_csv(
        OUTPUT_DIR / "metadata_differences.csv",
        index=False,
    )

    print("\nMetadata difference frequency:")

    for col in unique_columns:
        count = (result[col] > 1).sum()
        print(
            f"{col.replace('_UNIQUE_VALUES', '')}: "
            f"{count:,} groups"
        )

    return result


def investigate_high_repeat_satellites(records, repeated):
    print("\n========== HIGH REPEAT COUNT SATELLITES ==========")

    high_repeat = repeated[
        repeated["RECORD_COUNT"] >= HIGH_REPEAT_THRESHOLD
    ].copy()

    summary = (
        high_repeat.groupby("NORAD_CAT_ID")
        .agg(
            HIGH_REPEAT_EPOCHS=("RECORD_COUNT", "size"),
            MAX_RECORDS_AT_ONE_EPOCH=("RECORD_COUNT", "max"),
            TOTAL_RECORDS=("RECORD_COUNT", "sum"),
        )
        .sort_values(
            ["MAX_RECORDS_AT_ONE_EPOCH", "HIGH_REPEAT_EPOCHS"],
            ascending=False,
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "high_repeat_satellites.csv",
        index=False,
    )

    print(
        summary.head(20).to_string(index=False)
        if not summary.empty
        else "No high-repeat satellites found."
    )

    # Detailed records for NORAD 45755.
    focus = records[
        records["NORAD_CAT_ID"] == FOCUS_NORAD_ID
    ].copy()

    if not focus.empty:
        focus = focus.sort_values(
            ["_EPOCH", "_GP_ID_NUMERIC"],
            na_position="last",
        )

        focus.to_csv(
            OUTPUT_DIR / f"norad_{FOCUS_NORAD_ID}_repeated_records.csv",
            index=False,
        )

        print(
            f"\nNORAD {FOCUS_NORAD_ID}: "
            f"{len(focus):,} records in repeated groups"
        )

        print(
            focus.groupby("_EPOCH")
            .size()
            .sort_values(ascending=False)
            .head(15)
            .to_string()
        )
    else:
        print(f"NORAD {FOCUS_NORAD_ID} has no repeated groups.")

    return summary


def investigate_decay_proximity(records):
    print("\n========== DECAY PROXIMITY ANALYSIS ==========")

    available = [
        col for col in DECAY_COLUMNS
        if col in records.columns
    ]

    if not available:
        print("No decay date columns available.")
        return

    for col in available:
        records[f"_{col}_PARSED"] = pd.to_datetime(
            records[col],
            errors="coerce",
            utc=True,
        )

        valid_count = records[f"_{col}_PARSED"].notna().sum()

        print(f"{col}: {valid_count:,} valid dates")

        if valid_count == 0:
            continue

        records[f"DAYS_TO_{col}"] = (
            records[f"_{col}_PARSED"] - records["_EPOCH"]
        ).dt.total_seconds() / 86400

    # Investigate groups with at least 3 records at one epoch.
    high_repeat = records[
        records["RECORD_COUNT"] >= HIGH_REPEAT_THRESHOLD
    ].copy()

    for col in available:
        days_col = f"DAYS_TO_{col}"

        if days_col not in high_repeat.columns:
            continue

        proximity = high_repeat.dropna(
            subset=[days_col]
        ).copy()

        if proximity.empty:
            print(f"\nNo usable proximity data for {col}.")
            continue

        proximity.to_csv(
            OUTPUT_DIR / f"high_repeat_decay_{col.lower()}.csv",
            index=False,
        )

        summary = (
            proximity.groupby(
                ["NORAD_CAT_ID", "_EPOCH", "RECORD_COUNT"],
                as_index=False,
            )
            .agg(
                DAYS_TO_DECAY=(days_col, "median"),
            )
            .sort_values("DAYS_TO_DECAY")
        )

        summary.to_csv(
            OUTPUT_DIR / f"decay_proximity_summary_{col.lower()}.csv",
            index=False,
        )

        print(f"\n{col}:")
        print(f"High-repeat groups with dates: {len(summary):,}")
        print(
            summary.head(15).to_string(index=False)
        )


def main():
    print("Starting repeated epoch analysis...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    records, repeated = get_repeated_records(df)

    chronology = investigate_chronology(records)

    orbital = investigate_orbital_differences(records)

    metadata = investigate_metadata(records)

    investigate_high_repeat_satellites(records, repeated)

    investigate_decay_proximity(records)

    print("\n========== ANALYSIS COMPLETED ==========")
    print(f"Repeated groups: {len(repeated):,}")
    print(f"Chronology groups analysed: {len(chronology):,}")
    print(f"Orbital difference groups: {len(orbital):,}")
    print(f"Metadata groups analysed: {len(metadata):,}")
    print(f"Reports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
