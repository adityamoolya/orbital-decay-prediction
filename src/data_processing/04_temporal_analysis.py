
"""
Temporal analysis of historical satellite orbital data.

Analyses:
- Number of TLE observations per year
- Number of unique satellites per year
- Observation frequency per satellite
- First and last observation epoch per satellite

Outputs:
- observations_per_year.csv
- satellites_per_year.csv
- annual_summary.csv
- observations_per_satellite.csv
- observations_per_year.png
- satellites_per_year.png

Input:
    data/curated/tle_features.parquet

Output:
    reports/eda/temporal/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_FILE = Path("data/curated/tle_features.parquet")

OUTPUT_DIR = Path("reports/eda/temporal")
PLOT_DIR = OUTPUT_DIR / "plots"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_dataset():
    print("Loading dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    required_columns = ["EPOCH", "NORAD_CAT_ID"]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Records loaded: {len(df):,}")

    return df


# --------------------------------------------------
# PREPARE TEMPORAL DATA
# --------------------------------------------------

def prepare_data(df):
    print("\nPreparing temporal data...")

    analysis_df = df[["EPOCH", "NORAD_CAT_ID"]].copy()

    analysis_df["EPOCH"] = pd.to_datetime(
        analysis_df["EPOCH"],
        errors="coerce",
        utc=True
    )

    analysis_df["NORAD_CAT_ID"] = pd.to_numeric(
        analysis_df["NORAD_CAT_ID"],
        errors="coerce"
    )

    invalid_epochs = analysis_df["EPOCH"].isna().sum()
    invalid_ids = analysis_df["NORAD_CAT_ID"].isna().sum()

    print(f"Invalid epochs: {invalid_epochs:,}")
    print(f"Invalid NORAD IDs: {invalid_ids:,}")

    analysis_df = analysis_df.dropna(
        subset=["EPOCH", "NORAD_CAT_ID"]
    ).copy()

    analysis_df["YEAR"] = analysis_df["EPOCH"].dt.year

    print(f"Valid records: {len(analysis_df):,}")

    return analysis_df


# --------------------------------------------------
# ANNUAL OBSERVATION ANALYSIS
# --------------------------------------------------

def analyse_observations_per_year(df):
    print("\n========== OBSERVATIONS PER YEAR ==========")

    observations = (
        df.groupby("YEAR")
        .size()
        .rename("OBSERVATION_COUNT")
        .reset_index()
    )

    print(observations.to_string(index=False))

    return observations


# --------------------------------------------------
# UNIQUE SATELLITES PER YEAR
# --------------------------------------------------

def analyse_satellites_per_year(df):
    print("\n========== UNIQUE SATELLITES PER YEAR ==========")

    satellites = (
        df.groupby("YEAR")["NORAD_CAT_ID"]
        .nunique()
        .rename("UNIQUE_SATELLITES")
        .reset_index()
    )

    print(satellites.to_string(index=False))

    return satellites


# --------------------------------------------------
# SATELLITE-LEVEL ANALYSIS
# --------------------------------------------------

def analyse_satellite_history(df):
    print("\n========== SATELLITE OBSERVATION ANALYSIS ==========")

    results = []

    for norad_id, group in df.groupby("NORAD_CAT_ID"):

        group = group.sort_values("EPOCH")

        epochs = group["EPOCH"]

        gaps = epochs.diff().dt.total_seconds() / 86400

        results.append({
            "NORAD_CAT_ID": int(norad_id),
            "OBSERVATION_COUNT": len(group),
            "FIRST_EPOCH": epochs.min(),
            "LAST_EPOCH": epochs.max(),
            "ACTIVE_YEARS": group["YEAR"].nunique(),
            "MEDIAN_GAP_DAYS": gaps.median(),
            "MEAN_GAP_DAYS": gaps.mean(),
        })

    satellite_stats = pd.DataFrame(results)

    satellite_stats = satellite_stats.sort_values(
        "OBSERVATION_COUNT",
        ascending=False
    )

    print(f"Satellites analysed: {len(satellite_stats)}")

    print("\nTop 10 satellites by observation count:")
    print(satellite_stats.head(10).to_string(index=False))

    return satellite_stats


# --------------------------------------------------
# ANNUAL SUMMARY
# --------------------------------------------------

def create_annual_summary(observations, satellites):

    annual_summary = observations.merge(
        satellites,
        on="YEAR",
        how="outer"
    ).sort_values("YEAR")

    return annual_summary


# --------------------------------------------------
# SAVE CSV REPORTS
# --------------------------------------------------

def save_reports(
    observations,
    satellites,
    annual_summary,
    satellite_stats
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    observations.to_csv(
        OUTPUT_DIR / "observations_per_year.csv",
        index=False
    )

    satellites.to_csv(
        OUTPUT_DIR / "satellites_per_year.csv",
        index=False
    )

    annual_summary.to_csv(
        OUTPUT_DIR / "annual_summary.csv",
        index=False
    )

    satellite_stats.to_csv(
        OUTPUT_DIR / "observations_per_satellite.csv",
        index=False
    )

    print("\nCSV reports saved.")


# --------------------------------------------------
# VISUALISATIONS
# --------------------------------------------------

def plot_observations_per_year(observations):

    plt.figure(figsize=(12, 6))

    plt.bar(
        observations["YEAR"],
        observations["OBSERVATION_COUNT"]
    )

    plt.title("Satellite TLE Observations Per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Observations")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "observations_per_year.png",
        dpi=300
    )

    plt.close()


def plot_satellites_per_year(satellites):

    plt.figure(figsize=(12, 6))

    plt.plot(
        satellites["YEAR"],
        satellites["UNIQUE_SATELLITES"],
        marker="o"
    )

    plt.title("Unique Satellites Observed Per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Unique Satellites")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "satellites_per_year.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Starting temporal analysis...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    analysis_df = prepare_data(df)

    observations = analyse_observations_per_year(
        analysis_df
    )

    satellites = analyse_satellites_per_year(
        analysis_df
    )

    satellite_stats = analyse_satellite_history(
        analysis_df
    )

    annual_summary = create_annual_summary(
        observations,
        satellites
    )

    save_reports(
        observations,
        satellites,
        annual_summary,
        satellite_stats
    )

    print("\nGenerating temporal plots...")

    plot_observations_per_year(observations)
    plot_satellites_per_year(satellites)

    print("\n========== TEMPORAL ANALYSIS COMPLETED ==========")
    print(f"Reports saved in: {OUTPUT_DIR}")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
