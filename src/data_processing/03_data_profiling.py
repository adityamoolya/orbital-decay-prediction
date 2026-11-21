
"""
03_data_profiling.py

Exploratory Data Analysis (EDA) for the preprocessed TLE dataset.

Generates:
- Dataset overview
- Column-level profiling
- Numerical descriptive statistics
- Missing-value report
- Categorical value summaries
- Numerical distribution plots
- Correlation matrix
- Epoch coverage plot

Does not modify the source dataset.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/curated/tle_features.parquet"
)

OUTPUT_DIR = Path("reports/eda")

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_parquet(INPUT_FILE)


# ============================================================
# TEMPORARY DATA TYPE CONVERSION FOR ANALYSIS
# ============================================================

analysis_df = df.copy()

numeric_columns = [
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS",
    "PERIOD",
    "APOAPSIS",
    "PERIAPSIS",
]

for col in numeric_columns:
    analysis_df[col] = pd.to_numeric(
        analysis_df[col],
        errors="coerce"
    )

analysis_df["EPOCH"] = pd.to_datetime(
    analysis_df["EPOCH"],
    errors="coerce",
    utc=True
)

print("\n========== CONVERTED DATA TYPES ==========")
print(analysis_df[numeric_columns].dtypes)

print("\n========== ORBITAL STATISTICS ==========")
print(analysis_df[numeric_columns].describe().T)

print("\n========== CONSTANT COLUMNS ==========")
for col in analysis_df.columns:
    if analysis_df[col].nunique(dropna=False) == 1:
        print(f"{col}: {analysis_df[col].iloc[0]}")

print("\n========== NUMERIC CONVERSION FAILURES ==========")
for col in numeric_columns:
    failures = (
        df[col].notna()
        & analysis_df[col].isna()
    ).sum()

    if failures:
        print(f"{col}: {failures} conversion failures")


print("\n========== DATASET OVERVIEW ==========")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# ============================================================
# COLUMN PROFILING
# ============================================================

profile = pd.DataFrame({
    "data_type": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "missing_percentage": df.isna().mean() * 100,
    "unique_values": df.nunique(dropna=True),
})

profile["non_missing_count"] = len(df) - profile["missing_count"]

profile = profile[
    [
        "data_type",
        "non_missing_count",
        "missing_count",
        "missing_percentage",
        "unique_values",
    ]
]

profile.to_csv(OUTPUT_DIR / "column_profile.csv")

print("\n========== COLUMN PROFILE ==========")
print(profile.to_string())


# ============================================================
# NUMERICAL STATISTICS
# ============================================================

numeric_df = analysis_df.select_dtypes(include="number")

describe = numeric_df.describe().T

describe.to_csv(OUTPUT_DIR / "numerical_statistics.csv")

print("\n========== NUMERICAL STATISTICS ==========")
print(describe.to_string())


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns

with open(OUTPUT_DIR / "categorical_summary.txt", "w") as file:

    for col in categorical_columns:

        file.write(f"\n{'=' * 60}\n")
        file.write(f"COLUMN: {col}\n")
        file.write(f"{'=' * 60}\n")

        file.write(
            df[col]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )

        file.write("\n")


# ============================================================
# NUMERICAL DISTRIBUTIONS
# ============================================================

# Exclude identifier-like columns from distribution plots.
identifier_columns = {
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "GP_ID",
    "REV_AT_EPOCH",
}

plot_columns = [
    col for col in numeric_df.columns
    if col not in identifier_columns
]

print("\nGenerating numerical distribution plots...")

for col in plot_columns:

    values = numeric_df[col].dropna()

    if values.empty:
        continue

    plt.figure(figsize=(9, 5))

    plt.hist(values, bins=60)

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{col}_distribution.png")
    plt.close()


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

correlation_columns = [
    col for col in plot_columns
    if numeric_df[col].nunique() > 1
]

if len(correlation_columns) >= 2:

    correlation = numeric_df[correlation_columns].corr()

    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(14, 11))

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=90,
        fontsize=8,
    )

    ax.set_yticklabels(
        correlation.columns,
        fontsize=8,
    )

    fig.colorbar(image, ax=ax, label="Correlation")

    ax.set_title("Numerical Feature Correlation")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_matrix.png")
    plt.close()


# ============================================================
# EPOCH ANALYSIS
# ============================================================

if "EPOCH" in df.columns:

    # Temporary conversion: original EPOCH column remains unchanged.
    epoch = pd.to_datetime(df["EPOCH"], errors="coerce", utc=True)

    print("\n========== EPOCH COVERAGE ==========")
    print("Valid epochs:", epoch.notna().sum())
    print("Invalid epochs:", epoch.isna().sum())

    print("Earliest epoch:", epoch.min())
    print("Latest epoch:", epoch.max())

    valid_epochs = epoch.dropna()

    if not valid_epochs.empty:

        plt.figure(figsize=(12, 5))

        plt.hist(valid_epochs, bins=100)

        plt.title("Distribution of TLE Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Record Count")

        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(PLOT_DIR / "epoch_distribution.png")
        plt.close()


# ============================================================
# COMPLETION
# ============================================================

print("\n========== EDA COMPLETED ==========")
print(f"Reports saved in: {OUTPUT_DIR}")
print(f"Plots saved in: {PLOT_DIR}")
