
"""
Satellite history analysis.

Analyses the historical orbital evolution of selected satellites.

Parameters plotted:
- Perigee altitude
- Apogee altitude
- Mean motion
- Eccentricity
- BSTAR

Input:
    data/curated/tle_features.parquet

Output:
    reports/eda/satellite_history/
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path("data/curated/tle_features.parquet")

OUTPUT_DIR = Path("reports/eda/satellite_history")

# Satellites selected from temporal analysis
SELECTED_SATELLITES = [7003, 4133]

PARAMETERS = {
    "PERIAPSIS": "Perigee Altitude (km)",
    "APOAPSIS": "Apogee Altitude (km)",
    "MEAN_MOTION": "Mean Motion (revolutions/day)",
    "ECCENTRICITY": "Eccentricity",
    "BSTAR": "BSTAR",
}


# --------------------------------------------------
# LOAD DATA
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

    print("\nPreparing satellite history data...")

    required_columns = [
        "EPOCH",
        "NORAD_CAT_ID",
        *PARAMETERS.keys()
    ]

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

    analysis_df["NORAD_CAT_ID"] = pd.to_numeric(
        analysis_df["NORAD_CAT_ID"],
        errors="coerce"
    )

    for col in PARAMETERS:

        analysis_df[col] = pd.to_numeric(
            analysis_df[col],
            errors="coerce"
        )

    analysis_df = analysis_df.dropna(
        subset=["EPOCH", "NORAD_CAT_ID"]
    )

    analysis_df["NORAD_CAT_ID"] = (
        analysis_df["NORAD_CAT_ID"].astype(int)
    )

    print(f"Valid records: {len(analysis_df):,}")

    return analysis_df


# --------------------------------------------------
# FILTER SATELLITE
# --------------------------------------------------

def get_satellite_data(df, norad_id):

    satellite_df = df[
        df["NORAD_CAT_ID"] == norad_id
    ].copy()

    satellite_df = satellite_df.sort_values("EPOCH")

    if satellite_df.empty:
        print(f"No records found for NORAD {norad_id}")
        return None

    return satellite_df


# --------------------------------------------------
# SATELLITE SUMMARY
# --------------------------------------------------

def generate_summary(satellite_df, norad_id):

    return {
        "NORAD_CAT_ID": norad_id,
        "OBSERVATION_COUNT": len(satellite_df),
        "FIRST_EPOCH": satellite_df["EPOCH"].min(),
        "LAST_EPOCH": satellite_df["EPOCH"].max(),
        "ACTIVE_YEARS": satellite_df["EPOCH"].dt.year.nunique(),
    }


# --------------------------------------------------
# PLOT ORBITAL PARAMETER
# --------------------------------------------------

def plot_parameter(
    satellite_df,
    norad_id,
    parameter,
    ylabel,
    output_dir
):

    plot_df = satellite_df[
        ["EPOCH", parameter]
    ].dropna()

    if plot_df.empty:
        print(
            f"No valid values for {parameter} "
            f"(NORAD {norad_id})"
        )
        return

    plt.figure(figsize=(13, 6))

    plt.plot(
        plot_df["EPOCH"],
        plot_df[parameter],
        linewidth=0.8
    )

    plt.title(
        f"{ylabel} vs Time - NORAD {norad_id}"
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.gca().xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    plt.gca().xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            plt.gca().xaxis.get_major_locator()
        )
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    filename = f"{parameter.lower()}_vs_time.png"

    plt.savefig(
        output_dir / filename,
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# PROCESS SELECTED SATELLITES
# --------------------------------------------------

def analyse_satellite(df, norad_id):

    print(f"\n========== NORAD {norad_id} ==========")

    satellite_df = get_satellite_data(
        df,
        norad_id
    )

    if satellite_df is None:
        return None

    print(f"Observations: {len(satellite_df):,}")
    print(f"First epoch: {satellite_df['EPOCH'].min()}")
    print(f"Last epoch: {satellite_df['EPOCH'].max()}")

    output_dir = OUTPUT_DIR / f"NORAD_{norad_id}"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for parameter, ylabel in PARAMETERS.items():

        print(f"Generating {parameter} plot...")

        plot_parameter(
            satellite_df,
            norad_id,
            parameter,
            ylabel,
            output_dir
        )

    return generate_summary(
        satellite_df,
        norad_id
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Starting satellite history analysis...\n")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_dataset()

    analysis_df = prepare_data(df)

    summaries = []

    for norad_id in SELECTED_SATELLITES:

        summary = analyse_satellite(
            analysis_df,
            norad_id
        )

        if summary is not None:
            summaries.append(summary)

    if summaries:

        summary_df = pd.DataFrame(summaries)

        summary_df.to_csv(
            OUTPUT_DIR / "satellite_history_summary.csv",
            index=False
        )

    print("\n========== ANALYSIS COMPLETED ==========")

    print(f"Reports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
