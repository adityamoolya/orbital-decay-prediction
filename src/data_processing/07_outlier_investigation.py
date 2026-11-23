
"""
Investigates extreme orbital observations identified during IQR analysis.

Examines:
- Highest eccentricity
- Highest and lowest BSTAR
- Highest apogee
- Lowest perigee
- Lowest and highest mean motion

Exports selected extreme observations with satellite IDs, names,
epochs, values, and IQR flags.
"""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/curated/tle_features.parquet")
FLAGGED_FILE = Path("reports/eda/outliers/flagged_observations.csv")
OUTPUT_DIR = Path("reports/eda/outliers/investigation")

TOP_N = 20

PARAMETERS = {
    "eccentricity_high": ("ECCENTRICITY", "largest"),
    "bstar_high": ("BSTAR", "largest"),
    "bstar_low": ("BSTAR", "smallest"),
    "apogee_high": ("APOAPSIS", "largest"),
    "perigee_low": ("PERIAPSIS", "smallest"),
    "mean_motion_high": ("MEAN_MOTION", "largest"),
    "mean_motion_low": ("MEAN_MOTION", "smallest"),
}


def load_data():
    print("Loading curated dataset...")
    df = pd.read_parquet(INPUT_FILE)

    print("Loading IQR flagged observations...")
    flagged = pd.read_csv(FLAGGED_FILE, low_memory=False)

    return df, flagged


def prepare_data(df, flagged):
    columns = [
        "NORAD_CAT_ID",
        "OBJECT_NAME",
        "EPOCH",
        "ECCENTRICITY",
        "BSTAR",
        "APOAPSIS",
        "PERIAPSIS",
        "MEAN_MOTION",
    ]

    analysis_df = df[columns].copy()

    numeric_columns = [
        "ECCENTRICITY",
        "BSTAR",
        "APOAPSIS",
        "PERIAPSIS",
        "MEAN_MOTION",
    ]

    for col in numeric_columns:
        analysis_df[col] = pd.to_numeric(
            analysis_df[col], errors="coerce"
        )

    analysis_df["EPOCH"] = pd.to_datetime(
        analysis_df["EPOCH"], errors="coerce", utc=True
    )

    flagged_columns = [
        "NORAD_CAT_ID",
        "OBJECT_NAME",
        "EPOCH",
        "ECCENTRICITY_OUTLIER",
        "BSTAR_OUTLIER",
        "APOAPSIS_OUTLIER",
        "PERIAPSIS_OUTLIER",
        "MEAN_MOTION_OUTLIER",
    ]

    available_flags = [
        col for col in flagged_columns if col in flagged.columns
    ]

    flagged_subset = flagged[available_flags].copy()

    flagged_subset["EPOCH"] = pd.to_datetime(
        flagged_subset["EPOCH"], errors="coerce", utc=True
    )

    # Match records using satellite ID and epoch.
    # Duplicate epochs can exist, so merge may return multiple matches.
    analysis_df = analysis_df.merge(
        flagged_subset,
        on=["NORAD_CAT_ID", "OBJECT_NAME", "EPOCH"],
        how="left",
    )

    flag_columns = [
        col for col in available_flags if col.endswith("_OUTLIER")
    ]

    for col in flag_columns:
        analysis_df[col] = analysis_df[col].fillna(False).astype(bool)

    return analysis_df


def investigate_extremes(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for report_name, (parameter, direction) in PARAMETERS.items():
        print(f"\nInvestigating {parameter}: {direction}")

        ascending = direction == "smallest"

        result = (
            df.dropna(subset=[parameter])
            .sort_values(parameter, ascending=ascending)
            .head(TOP_N)
            .copy()
        )

        flag_column = f"{parameter}_OUTLIER"

        if flag_column not in result.columns:
            result[flag_column] = False

        output_file = OUTPUT_DIR / f"{report_name}.csv"
        result.to_csv(output_file, index=False)

        print(result[
            [
                "NORAD_CAT_ID",
                "OBJECT_NAME",
                "EPOCH",
                parameter,
                flag_column,
            ]
        ].to_string(index=False))

        print(f"Saved: {output_file}")


def main():
    print("Starting extreme observation investigation...\n")

    df, flagged = load_data()

    print(f"Curated records: {len(df):,}")
    print(f"Flagged records: {len(flagged):,}")

    analysis_df = prepare_data(df, flagged)

    investigate_extremes(analysis_df)

    print("\n========== INVESTIGATION COMPLETED ==========")
    print(f"Reports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
