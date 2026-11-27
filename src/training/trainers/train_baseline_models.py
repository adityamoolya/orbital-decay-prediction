"""
train_baseline_models.py

Master baseline model training pipeline:
  1. Loads features.parquet and locked test satellite ID list.
  2. Partitions into train and test sets by norad_id membership.
  3. Confirms ZERO satellite overlap between train and test (hard-fails if overlap exists).
  4. Drops initial boundary NaN rows with full diagnostic logging.
  5. Trains all three baseline models sequentially:
       - Linear Regression (floor)
       - Random Forest Regressor (n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
       - Hist Gradient Boosting Regressor (random_state=42)
  6. Evaluates test MAE and RMSE for each model.
  7. Records and prints wall-clock training times.
  8. Saves predicted-vs-actual scatter plots to data/plots/ and reports/modeling/.
  9. Saves all three trained model artifacts (.joblib) to models/.
  10. Prints a unified performance and timing comparison table.

Run via:
  python src/training/trainers/train_baseline_models.py
"""

from pathlib import Path
import sys
import time
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Add src/training to import data_loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import load_train_test_data, save_scatter_plot

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


def main():
    print("=" * 80)
    print("SATELLITE DECAY PREDICTION: BASELINE MODEL TRAINING PIPELINE")
    print("=" * 80)

    # 1. Load data & verify zero overlap
    X_train, y_train, X_test, y_test, test_df = load_train_test_data()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Model definitions
    models_config = [
        {
            "name": "Linear Regression (Floor)",
            "key": "linear_regression",
            "model": LinearRegression(),
            "save_path": MODELS_DIR / "linear_regression.joblib",
            "plot_paths": [
                BASE_DIR / "data" / "plots" / "predicted_vs_actual_linear_regression.png",
                BASE_DIR / "reports" / "modeling" / "predicted_vs_actual_linear_regression.png",
            ],
        },
        {
            "name": "Random Forest Regressor",
            "key": "random_forest",
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            ),
            "save_path": MODELS_DIR / "random_forest.joblib",
            "plot_paths": [
                BASE_DIR / "data" / "plots" / "predicted_vs_actual_random_forest.png",
                BASE_DIR / "reports" / "modeling" / "predicted_vs_actual_random_forest.png",
            ],
        },
        {
            "name": "Hist Gradient Boosting",
            "key": "hist_gradient_boosting",
            "model": HistGradientBoostingRegressor(
                random_state=42,
                verbose=1,
            ),
            "save_path": MODELS_DIR / "hist_gradient_boosting.joblib",
            "plot_paths": [
                BASE_DIR / "data" / "plots" / "predicted_vs_actual_hist_gradient_boosting.png",
                BASE_DIR / "reports" / "modeling" / "predicted_vs_actual_hist_gradient_boosting.png",
            ],
        },
    ]

    results = []

    # 2. Iterate through models
    for cfg in models_config:
        name = cfg["name"]
        model = cfg["model"]

        print(f"\n{'=' * 80}")
        print(f"TRAINING: {name.upper()}")
        print("=" * 80)

        t_start = time.time()
        model.fit(X_train, y_train)
        wall_clock = time.time() - t_start

        print(f"\n  ✓ Training wall-clock time: {wall_clock:.2f} seconds ({wall_clock / 60:.2f} min)")

        # Evaluate on test set
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        print(f"  ✓ Evaluation on {len(y_test):,} test observations:")
        print(f"      MAE:  {mae:.2f} days")
        print(f"      RMSE: {rmse:.2f} days")

        # Save model
        joblib.dump(model, cfg["save_path"])
        print(f"  ✓ Saved model artifact to: {cfg['save_path']}")

        # Save plot
        save_scatter_plot(
            y_test=y_test,
            y_pred=y_pred,
            model_name=name,
            output_paths=cfg["plot_paths"],
            metrics={"MAE": mae, "RMSE": rmse},
        )

        results.append({
            "Model": name,
            "MAE (days)": round(mae, 2),
            "RMSE (days)": round(rmse, 2),
            "Training Time (s)": round(wall_clock, 2),
            "Training Time (min)": round(wall_clock / 60, 2),
        })

    # 3. Print unified comparison table
    print(f"\n{'=' * 80}")
    print("BASELINE MODEL COMPARISON SUMMARY (EVALUATED ON UNSEEN TEST SATELLITES)")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    summary_csv = BASE_DIR / "reports" / "modeling" / "baseline_model_comparison.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Comparison table saved to: {summary_csv}")


if __name__ == "__main__":
    main()
