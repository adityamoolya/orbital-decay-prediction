"""
data_loader.py

Shared data loading, split verification, and preprocessing utility for baseline models.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "modeling" / "features.parquet"
SPLITS_DIR = BASE_DIR / "data" / "splits"
TEST_SATS_CSV = SPLITS_DIR / "test_satellites.csv"
TRAIN_SATS_CSV = SPLITS_DIR / "train_satellites.csv"

# The 30 tabular predictive features
FEATURE_COLS = [
    # Native orbital parameters
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS",
    "PERIOD",
    "APOAPSIS",
    "PERIAPSIS",
    # Dynamic rates of change
    "mean_motion_rate",
    "bstar_rate",
    "eccentricity_rate",
    "periapsis_rate",
    # Rolling averages (5, 10, 20 TLEs)
    "mean_motion_roll5",
    "bstar_roll5",
    "eccentricity_roll5",
    "periapsis_roll5",
    "mean_motion_roll10",
    "bstar_roll10",
    "eccentricity_roll10",
    "periapsis_roll10",
    "mean_motion_roll20",
    "bstar_roll20",
    "eccentricity_roll20",
    "periapsis_roll20",
    # Rolling volatility
    "bstar_std5",
    "bstar_std10",
    "bstar_std20",
    # Trajectory lifecycle index
    "lifetime_fraction",
]

TARGET_COL = "days_to_decay"


def load_train_test_data():
    """
    Loads features.parquet and locked satellite ID lists.
    Enforces strict zero satellite overlap.
    Drops initial boundary NaNs with full transparency.
    Returns: X_train, y_train, X_test, y_test, test_df
    """
    print("=" * 70)
    print("LOADING DATASET AND APPLYING SATELLITE-LEVEL PARTITION")
    print("=" * 70)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found: {FEATURES_PATH}")
    if not TEST_SATS_CSV.exists():
        raise FileNotFoundError(
            f"Test satellite file not found: {TEST_SATS_CSV}\n"
            "Please run src/modeling/03_train_test_split.py first to create the locked split."
        )

    df = pd.read_parquet(FEATURES_PATH)
    test_sats = pd.read_csv(TEST_SATS_CSV)["NORAD_CAT_ID"].values

    if TRAIN_SATS_CSV.exists():
        train_sats = pd.read_csv(TRAIN_SATS_CSV)["NORAD_CAT_ID"].values
    else:
        all_sats = df["NORAD_CAT_ID"].unique()
        train_sats = np.array([s for s in all_sats if s not in test_sats])

    # 1. HARD CHECK: Zero satellite overlap verification
    overlap = set(train_sats).intersection(set(test_sats))
    if len(overlap) > 0:
        raise ValueError(
            f"CRITICAL ERROR: Detected {len(overlap)} overlapping satellites between train and test: {overlap}!"
        )

    print(f"  Satellite Split: {len(train_sats)} train satellites | {len(test_sats)} test satellites")
    print(f"  ✓ Zero satellite overlap confirmed (0 satellites in common).")

    # 2. Handle NaN rows from rolling/diff boundary window
    nan_mask = df[FEATURE_COLS].isna().any(axis=1)
    dropped_count = int(nan_mask.sum())
    affected_sats = int(df.loc[nan_mask, "NORAD_CAT_ID"].nunique())

    print(f"\n  Boundary NaN Handling:")
    print(f"    Total raw observations: {len(df):,}")
    print(f"    Dropped {dropped_count:,} rows with NaNs ({dropped_count / len(df) * 100:.3f}%) across {affected_sats} satellites.")
    print("    Rationale: Dropped rather than imputed because the initial observation of a satellite")
    print("    has no trailing window for rolling trends or prior epoch for numerical differencing.")
    print("    Imputing values at t_0 would invent artificial signal that does not exist.")

    df_clean = df[~nan_mask].copy()

    # 3. Partition train and test sets by norad_id membership
    train_df = df_clean[df_clean["NORAD_CAT_ID"].isin(train_sats)].copy()
    test_df = df_clean[df_clean["NORAD_CAT_ID"].isin(test_sats)].copy()

    # Final hard check on partition rows
    partition_overlap = set(train_df["NORAD_CAT_ID"]).intersection(set(test_df["NORAD_CAT_ID"]))
    if len(partition_overlap) > 0:
        raise ValueError(f"CRITICAL ERROR: Overlapping satellite IDs found in partition rows: {partition_overlap}")

    print(f"\n  Partition Summary:")
    print(f"    Train: {len(train_df):,} rows ({train_df['NORAD_CAT_ID'].nunique()} satellites)")
    print(f"    Test:  {len(test_df):,} rows ({test_df['NORAD_CAT_ID'].nunique()} satellites)")
    print(f"    Feature columns: {len(FEATURE_COLS)}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    return X_train, y_train, X_test, y_test, test_df


def save_scatter_plot(y_test, y_pred, model_name, output_paths, metrics=None):
    """
    Generates a predicted vs actual scatter plot on the test set.
    Saves to each path in output_paths.
    """
    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7), dpi=150)

    # Subsample if dataset is large so scatter plot renders quickly and crisply
    max_scatter_points = 25000
    if len(y_test) > max_scatter_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_test), size=max_scatter_points, replace=False)
        y_test_plot = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]
        y_pred_plot = y_pred[idx]
        subsample_note = f" (Subsampled {max_scatter_points:,} / {len(y_test):,} test points)"
    else:
        y_test_plot = y_test
        y_pred_plot = y_pred
        subsample_note = ""

    plt.scatter(y_test_plot, y_pred_plot, alpha=0.25, s=8, color="#1f77b4", edgecolors="none")

    # Reference y = x line
    min_val = min(float(y_test.min()), float(y_pred.min()), 0)
    max_val = max(float(y_test.max()), float(y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal 1:1 line (y = x)")

    plt.xlabel("Actual days_to_decay (days)", fontsize=11)
    plt.ylabel("Predicted days_to_decay (days)", fontsize=11)
    plt.title(f"{model_name}: Predicted vs. Actual days_to_decay{subsample_note}", fontsize=12, pad=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    if metrics:
        textstr = f"MAE:  {metrics['MAE']:.2f} days\nRMSE: {metrics['RMSE']:.2f} days"
        props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc")
        plt.gca().text(0.05, 0.92, textstr, transform=plt.gca().transAxes, fontsize=11,
                       verticalalignment="top", bbox=props)

    plt.legend(loc="lower right")
    plt.tight_layout()

    for p in output_paths:
        plt.savefig(p)
        print(f"  ✓ Saved scatter plot to: {p}")

    plt.close()
