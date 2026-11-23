
"""
Outlier analysis of satellite orbital data.

Analyses:
- Eccentricity
- BSTAR
- Apogee altitude
- Perigee altitude
- Mean motion

Method:
- Interquartile Range (IQR)
- 1.5 * IQR lower and upper boundaries

Outputs:
- outlier_summary.csv
- flagged_observations.csv
- Individual boxplots

Input:
    data/curated/tle_features.parquet

Output:
    reports/eda/outliers/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_FILE = Path("data/curated/tle_features.parquet")

OUTPUT_DIR = Path("reports/eda/outliers")
PLOT_DIR = OUTPUT_DIR / "plots"


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PARAMETERS = [
    "ECCENTRICITY",
    "BSTAR",
    "APOAPSIS",
    "PERIAPSIS",
    "MEAN_MOTION",
]

IDENTIFIER_COLUMNS = [
    "NORAD_CAT_ID",
    "OBJECT_NAME",
    "EPOCH",
]


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():

    print("Loading dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Records loaded: {len(df):,}")

    return df


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

def prepare_data(df):

    print("\nPreparing data for outlier analysis...")

    required_columns = IDENTIFIER_COLUMNS + PARAMETERS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    analysis_df = df[required_columns].copy()

    analysis_df["EPOCH"] = pd.to_datetime(
        analysis_df["EPOCH"],
        errors="coerce",
        utc=True
    )

    for col in PARAMETERS:

        analysis_df[col] = pd.to_numeric(
            analysis_df[col],
            errors="coerce"
        )

    print("\nNumeric conversion failures:")

    for col in PARAMETERS:

        failures = (
            analysis_df[col].isna().sum()
            - df[col].isna().sum()
        )

        print(f"{col}: {failures:,}")

    return analysis_df


# --------------------------------------------------
# IQR OUTLIER DETECTION
# --------------------------------------------------

def detect_outliers(df):

    print("\n========== IQR OUTLIER ANALYSIS ==========")

    summary = []

    flagged_df = df.copy()

    for col in PARAMETERS:

        values = df[col].dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        lower_mask = df[col] < lower_bound
        upper_mask = df[col] > upper_bound

        outlier_mask = lower_mask | upper_mask

        lower_count = int(lower_mask.sum())
        upper_count = int(upper_mask.sum())

        outlier_count = int(outlier_mask.sum())

        outlier_percentage = (
            outlier_count / values.count()
        ) * 100 if values.count() else 0

        flagged_df[f"{col}_OUTLIER"] = outlier_mask

        summary.append({
            "PARAMETER": col,
            "VALID_RECORDS": values.count(),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "LOWER_BOUND": lower_bound,
            "UPPER_BOUND": upper_bound,
            "LOWER_OUTLIERS": lower_count,
            "UPPER_OUTLIERS": upper_count,
            "TOTAL_OUTLIERS": outlier_count,
            "OUTLIER_PERCENTAGE": outlier_percentage,
        })

        print(f"\n{col}")
        print(f"Q1: {q1}")
        print(f"Q3: {q3}")
        print(f"IQR: {iqr}")
        print(f"Lower bound: {lower_bound}")
        print(f"Upper bound: {upper_bound}")
        print(f"Lower outliers: {lower_count:,}")
        print(f"Upper outliers: {upper_count:,}")
        print(f"Total outliers: {outlier_count:,}")
        print(f"Outlier percentage: {outlier_percentage:.4f}%")

    summary_df = pd.DataFrame(summary)

    # Keep observations flagged by at least one parameter.
    flag_columns = [
        f"{col}_OUTLIER" for col in PARAMETERS
    ]

    flagged_observations = flagged_df[
        flagged_df[flag_columns].any(axis=1)
    ].copy()

    return summary_df, flagged_observations


# --------------------------------------------------
# GENERATE BOXPLOTS
# --------------------------------------------------

def generate_boxplots(df, summary_df):

    print("\nGenerating boxplots...")

    for col in PARAMETERS:

        values = df[col].dropna()

        if values.empty:
            continue

        plt.figure(figsize=(10, 5))

        plt.boxplot(
            values,
            vert=False,
            showfliers=False
        )

        plt.title(f"Outlier Distribution: {col}")
        plt.xlabel(col)
        plt.grid(axis="x", alpha=0.3)

        outlier_count = summary_df.loc[
            summary_df["PARAMETER"] == col,
            "TOTAL_OUTLIERS"
        ].iloc[0]

        plt.text(
            0.98,
            0.95,
            f"IQR outliers: {outlier_count:,}",
            transform=plt.gca().transAxes,
            horizontalalignment="right",
            verticalalignment="top"
        )

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / f"{col.lower()}_outliers.png",
            dpi=300
        )

        plt.close()


# --------------------------------------------------
# SAVE REPORTS
# --------------------------------------------------

def save_reports(summary_df, flagged_observations):

    print("\nSaving reports...")

    summary_df.to_csv(
        OUTPUT_DIR / "outlier_summary.csv",
        index=False
    )

    flagged_observations.to_csv(
        OUTPUT_DIR / "flagged_observations.csv",
        index=False
    )

    print(
        f"Flagged observations: "
        f"{len(flagged_observations):,}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Starting outlier analysis...\n")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_dataset()

    analysis_df = prepare_data(df)

    summary_df, flagged_observations = detect_outliers(
        analysis_df
    )

    generate_boxplots(
        analysis_df,
        summary_df
    )

    save_reports(
        summary_df,
        flagged_observations
    )

    print("\n========== OUTLIER ANALYSIS COMPLETED ==========")

    print(f"Reports saved in: {OUTPUT_DIR}")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
