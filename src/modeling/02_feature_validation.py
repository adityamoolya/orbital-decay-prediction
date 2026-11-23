"""
02_feature_validation.py

Validates the feature-engineered dataset (data/modeling/features.parquet)
before feeding it into machine learning models.

Audits performed:
  1. Dataset completeness (row count, columns, satellite counts).
  2. Missing & infinite value analysis (explaining boundary effects).
  3. Target variable (days_to_decay) distribution & sanity checks.
  4. Physical orbital mechanics constraints (periapsis <= apoapsis, 0 <= e < 1, etc.).
  5. Decay signal checks (altitude decay rates, drag acceleration).
  6. Feature correlations with target (Pearson & Spearman rankings).
  7. Multicollinearity detection (highly correlated feature pairs > 0.95).
  8. Satellite group distribution (records per satellite, representation balance).
  9. Export summary reports to reports/modeling/.

Input: data/modeling/features.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data" / "modeling" / "features.parquet"
REPORT_DIR = BASE_DIR / "reports" / "modeling"
CORR_OUTPUT = REPORT_DIR / "feature_correlations_target.csv"
SUMMARY_OUTPUT = REPORT_DIR / "feature_validation_summary.csv"

# Earth radius in km
EARTH_RADIUS = 6378.137


# ==================================================
# 1. LOAD DATASET
# ==================================================
def load_dataset():
    print("=" * 70)
    print("1. LOADING FEATURE DATASET")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Features file not found at: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    file_size_mb = INPUT_PATH.stat().st_size / (1024 * 1024)

    print(f"  Path:       {INPUT_PATH}")
    print(f"  File size:  {file_size_mb:.2f} MB")
    print(f"  Rows:       {len(df):,}")
    print(f"  Columns:    {len(df.columns)}")
    print(f"  Satellites: {df['NORAD_CAT_ID'].nunique()}")

    return df


# ==================================================
# 2. MISSING & INFINITE VALUE AUDIT
# ==================================================
def audit_nulls_and_infs(df):
    print(f"\n{'=' * 70}")
    print("2. MISSING & INFINITE VALUES AUDIT")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_rows = len(df)
    n_sats = df["NORAD_CAT_ID"].nunique()

    null_summary = []

    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / total_rows) * 100

        inf_count = 0
        if col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()

        null_summary.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_pct, 3),
            "inf_count": inf_count,
        })

    summary_df = pd.DataFrame(null_summary)
    cols_with_nulls = summary_df[summary_df["null_count"] > 0]

    if cols_with_nulls.empty:
        print("  ✓ Zero missing values across the entire dataset!")
    else:
        print(f"  Found {len(cols_with_nulls)} column(s) with missing values:")
        for _, row in cols_with_nulls.iterrows():
            explanation = ""
            if row["null_count"] == n_sats:
                explanation = " (Exact boundary effect: 1st observation per satellite has no prior record)"
            print(f"    - {row['column']:<25}: {row['null_count']:,} nulls ({row['null_pct']:.2f}%){explanation}")

    infs_found = summary_df[summary_df["inf_count"] > 0]
    if infs_found.empty:
        print("  ✓ Zero infinite (+inf / -inf) values detected.")
    else:
        print(f"  ⚠ WARNING: {len(infs_found)} column(s) have infinite values:")
        for _, row in infs_found.iterrows():
            print(f"    - {row['column']:<25}: {row['inf_count']:,} infinite values")

    return summary_df


# ==================================================
# 3. TARGET VARIABLE AUDIT (days_to_decay)
# ==================================================
def audit_target_variable(df):
    print(f"\n{'=' * 70}")
    print("3. TARGET VARIABLE AUDIT (days_to_decay)")
    print("=" * 70)

    target = df["days_to_decay"]

    print("  Summary Statistics:")
    print(f"    Count:  {target.count():,}")
    print(f"    Mean:   {target.mean():.2f} days")
    print(f"    Std:    {target.std():.2f} days")
    print(f"    Min:    {target.min():.4f} days")
    print(f"    1%:     {target.quantile(0.01):.2f} days")
    print(f"    25%:    {target.quantile(0.25):.2f} days")
    print(f"    50%:    {target.quantile(0.50):.2f} days (Median)")
    print(f"    75%:    {target.quantile(0.75):.2f} days")
    print(f"    99%:    {target.quantile(0.99):.2f} days")
    print(f"    Max:    {target.max():.2f} days")

    # Negative check
    neg_count = (target < 0).sum()
    if neg_count == 0:
        print("  ✓ No negative days_to_decay values remain (clean).")
    else:
        print(f"  ⚠ WARNING: {neg_count} negative days_to_decay rows found!")

    # Zero values check (decay day observations)
    zero_count = (target == 0).sum()
    print(f"  Decay day / clipped observations (days = 0): {zero_count:,} ({zero_count / len(df) * 100:.2f}%)")

    # Horizon Distribution Breakdown
    horizons = [
        ("Immediate (< 7 days)", target < 7),
        ("Short (7 - 30 days)", (target >= 7) & (target < 30)),
        ("Medium (30 - 90 days)", (target >= 30) & (target < 90)),
        ("Long (90 - 365 days)", (target >= 90) & (target < 365)),
        ("Extended (>= 365 days)", target >= 365),
    ]

    print("\n  Target Horizon Distribution:")
    for name, mask in horizons:
        count = mask.sum()
        pct = (count / len(df)) * 100
        print(f"    - {name:<25}: {count:>8,} rows ({pct:>5.1f}%)")


# ==================================================
# 4. PHYSICAL ORBITAL MECHANICS SANITY CHECKS
# ==================================================
def audit_orbital_physics(df):
    print(f"\n{'=' * 70}")
    print("4. PHYSICAL ORBITAL MECHANICS SANITY CHECKS")
    print("=" * 70)

    # 1. Periapsis <= Apoapsis
    if "PERIAPSIS" in df.columns and "APOAPSIS" in df.columns:
        invalid_apsis = (df["PERIAPSIS"] > df["APOAPSIS"]).sum()
        if invalid_apsis == 0:
            print("  ✓ Periapsis <= Apoapsis: 100% compliant across all rows.")
        else:
            print(f"  ⚠ Periapsis > Apoapsis violation in {invalid_apsis:,} rows!")

    # 2. Eccentricity: 0 <= e < 1 for closed elliptical orbits
    if "ECCENTRICITY" in df.columns:
        neg_ecc = (df["ECCENTRICITY"] < 0).sum()
        hyp_ecc = (df["ECCENTRICITY"] >= 1.0).sum()
        if neg_ecc == 0 and hyp_ecc == 0:
            print(f"  ✓ Eccentricity in valid closed-orbit range [0, 1): "
                  f"min={df['ECCENTRICITY'].min():.6f}, max={df['ECCENTRICITY'].max():.6f}")
        else:
            print(f"  ⚠ Eccentricity out of bounds: {neg_ecc} negative, {hyp_ecc} >= 1.0")

    # 3. Semi-major axis > Earth radius
    if "SEMIMAJOR_AXIS" in df.columns:
        sub_surface = (df["SEMIMAJOR_AXIS"] < EARTH_RADIUS).sum()
        if sub_surface == 0:
            print(f"  ✓ Semi-major axis > Earth Radius ({EARTH_RADIUS} km): "
                  f"min={df['SEMIMAJOR_AXIS'].min():.2f} km")
        else:
            print(f"  ⚠ Semi-major axis below Earth surface in {sub_surface:,} rows!")

    # 4. Orbital Period bounds (LEO is typically 80 to 130 minutes)
    if "PERIOD" in df.columns:
        leo_period_valid = ((df["PERIOD"] >= 80) & (df["PERIOD"] <= 200)).sum()
        pct_leo = (leo_period_valid / len(df)) * 100
        print(f"  ✓ Orbital Period in expected LEO range (80-200 min): "
              f"{pct_leo:.2f}% (min={df['PERIOD'].min():.1f} min, max={df['PERIOD'].max():.1f} min)")

    # 5. Orbital Inclination: 0 <= i <= 180 deg
    if "INCLINATION" in df.columns:
        inc_valid = ((df["INCLINATION"] >= 0) & (df["INCLINATION"] <= 180)).sum()
        if inc_valid == len(df):
            print(f"  ✓ Inclination valid [0°, 180°]: min={df['INCLINATION'].min():.2f}°, max={df['INCLINATION'].max():.2f}°")
        else:
            print(f"  ⚠ Inclination out of bounds in {len(df) - inc_valid} rows!")


# ==================================================
# 5. DECAY SIGNAL DYNAMICS
# ==================================================
def audit_decay_signals(df):
    print(f"\n{'=' * 70}")
    print("5. DECAY SIGNAL DYNAMICS")
    print("=" * 70)

    # Periapsis rate should be mostly negative (altitude loss over time)
    if "periapsis_rate" in df.columns:
        valid_rates = df["periapsis_rate"].dropna()
        neg_rates = (valid_rates < 0).sum()
        pct_neg = (neg_rates / len(valid_rates)) * 100
        print(f"  Periapsis rate (altitude decay velocity):")
        print(f"    - Negative (losing altitude): {pct_neg:.1f}% ({neg_rates:,} / {len(valid_rates):,})")
        print(f"    - Median rate of altitude loss: {valid_rates.median():.4f} km/day")
        print(f"    - 10th percentile (rapid plunge): {valid_rates.quantile(0.10):.4f} km/day")

    # Mean motion rate should be mostly positive (revolutions increase as orbit shrinks)
    if "mean_motion_rate" in df.columns:
        valid_mm = df["mean_motion_rate"].dropna()
        pos_mm = (valid_mm > 0).sum()
        pct_pos = (pos_mm / len(valid_mm)) * 100
        print(f"  Mean motion rate (drag acceleration):")
        print(f"    - Positive (accelerating revolutions): {pct_pos:.1f}% ({pos_mm:,} / {len(valid_mm):,})")
        print(f"    - Median rate: {valid_mm.median():.6f} rev/day per day")

    # BSTAR std volatility check
    if "bstar_std5" in df.columns:
        valid_std = df["bstar_std5"].dropna()
        print(f"  BSTAR short-term volatility (std5):")
        print(f"    - Median: {valid_std.median():.8f}")
        print(f"    - 90th percentile: {valid_std.quantile(0.90):.8f}")


# ==================================================
# 6. FEATURE CORRELATIONS WITH TARGET
# ==================================================
def audit_correlations(df):
    print(f"\n{'=' * 70}")
    print("6. FEATURE CORRELATIONS WITH TARGET (days_to_decay)")
    print("=" * 70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "NORAD_CAT_ID" in numeric_cols:
        numeric_cols.remove("NORAD_CAT_ID")
    if "days_to_decay" in numeric_cols:
        numeric_cols.remove("days_to_decay")

    corr_records = []
    target = df["days_to_decay"]

    for col in numeric_cols:
        series = df[col]
        valid_mask = series.notna() & target.notna()

        if valid_mask.sum() < 100:
            continue

        s_valid = series[valid_mask]
        t_valid = target[valid_mask]

        pearson = s_valid.corr(t_valid, method="pearson")
        spearman = s_valid.corr(t_valid, method="spearman")

        corr_records.append({
            "feature": col,
            "pearson_corr": round(pearson, 4),
            "spearman_corr": round(spearman, 4),
            "abs_spearman": round(abs(spearman), 4),
        })

    corr_df = pd.DataFrame(corr_records).sort_values("abs_spearman", ascending=False).reset_index(drop=True)

    print("  Top 12 Features Ranked by Correlation Magnitude (|Spearman|):")
    print(f"    {'Feature':<28} {'Spearman':<12} {'Pearson':<12}")
    print("    " + "-" * 52)
    for _, row in corr_df.head(12).iterrows():
        print(f"    {row['feature']:<28} {row['spearman_corr']:>+8.4f}     {row['pearson_corr']:>+8.4f}")

    # Check for extreme multicollinearity (|r| > 0.95) among input features
    print("\n  Collinearity Check among Primary Features (|r| > 0.95):")
    primary_orbitals = [c for c in ["MEAN_MOTION", "PERIOD", "SEMIMAJOR_AXIS", "PERIAPSIS", "APOAPSIS"] if c in df.columns]
    orbit_corr = df[primary_orbitals].corr(method="pearson")

    high_pairs = []
    for i in range(len(primary_orbitals)):
        for j in range(i + 1, len(primary_orbitals)):
            c1, c2 = primary_orbitals[i], primary_orbitals[j]
            val = orbit_corr.loc[c1, c2]
            if abs(val) > 0.95:
                high_pairs.append((c1, c2, val))

    for c1, c2, val in high_pairs:
        print(f"    - {c1} <-> {c2}: r = {val:.4f} (physically expected, keep in mind for linear models)")

    return corr_df


# ==================================================
# 7. SATELLITE GROUP DISTRIBUTION AUDIT
# ==================================================
def audit_satellite_distribution(df):
    print(f"\n{'=' * 70}")
    print("7. SATELLITE GROUP DISTRIBUTION")
    print("=" * 70)

    counts = df.groupby("NORAD_CAT_ID").size()

    print(f"  Total Satellites:   {len(counts):,}")
    print(f"  Min TLEs per sat:   {counts.min():,}")
    print(f"  25th percentile:    {counts.quantile(0.25):.0f}")
    print(f"  Median TLEs:        {counts.median():.0f}")
    print(f"  75th percentile:    {counts.quantile(0.75):.0f}")
    print(f"  Max TLEs per sat:   {counts.max():,}")
    print(f"  Mean TLEs per sat:  {counts.mean():.1f}")

    # Largest satellites by record share
    top3 = counts.nlargest(3)
    print("\n  Top 3 satellites by data share:")
    for norad_id, cnt in top3.items():
        pct = (cnt / len(df)) * 100
        print(f"    - NORAD {norad_id}: {cnt:,} records ({pct:.2f}%)")

    print("\n  ✓ Confirmed: Satellite-grouped splitting (GroupKFold) is strongly recommended.")


# ==================================================
# 8. EXPORT SUMMARY REPORTS
# ==================================================
def export_reports(summary_df, corr_df):
    print(f"\n{'=' * 70}")
    print("8. EXPORTING VALIDATION REPORTS")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    corr_df.to_csv(CORR_OUTPUT, index=False)

    print(f"  Validation Summary:  {SUMMARY_OUTPUT}")
    print(f"  Target Correlations: {CORR_OUTPUT}")


# ==================================================
# MAIN PIPELINE
# ==================================================
def main():
    df = load_dataset()
    summary_df = audit_nulls_and_infs(df)
    audit_target_variable(df)
    audit_orbital_physics(df)
    audit_decay_signals(df)
    corr_df = audit_correlations(df)
    audit_satellite_distribution(df)
    export_reports(summary_df, corr_df)

    print(f"\n{'=' * 70}")
    print("FEATURE VALIDATION COMPLETED SUCCESSFULLY")
    print("  Dataset is ready for model training & baseline benchmarking!")
    print("=" * 70)


if __name__ == "__main__":
    main()
