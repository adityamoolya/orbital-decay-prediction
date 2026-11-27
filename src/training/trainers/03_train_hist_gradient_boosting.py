"""
03_train_hist_gradient_boosting.py

Trains and evaluates a Histogram-based Gradient Boosting Regressor for satellite decay prediction:
  - Architecture: HistGradientBoostingRegressor(random_state=42).
  - Fast, optimized CPU gradient boosting engineered for large tabular datasets (~800k rows).
  - Trains exclusively on train-satellite observations.
  - Evaluates exclusively on test-satellite observations.
  - Measures wall-clock training time.
  - Computes test set MAE and RMSE.
  - Generates predicted-vs-actual scatter plot.
  - Saves the trained model artifact to models/hist_gradient_boosting.joblib.

Run via:
  python src/training/trainers/03_train_hist_gradient_boosting.py
"""

from pathlib import Path
import sys
import time
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Add src/training to import data_loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import load_train_test_data, save_scatter_plot

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "hist_gradient_boosting.joblib"
PLOT_PATHS = [
    BASE_DIR / "data" / "plots" / "predicted_vs_actual_hist_gradient_boosting.png",
    BASE_DIR / "reports" / "modeling" / "predicted_vs_actual_hist_gradient_boosting.png",
]

RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("BASELINE MODEL 3: HISTOGRAM GRADIENT BOOSTING REGRESSOR")
    print("=" * 70)

    # 1. Load data
    X_train, y_train, X_test, y_test, _ = load_train_test_data()

    # 2. Train model with wall-clock timing
    print("\n" + "=" * 70)
    print("TRAINING HISTOGRAM GRADIENT BOOSTING REGRESSOR")
    print(f"  Configuration: HistGradientBoostingRegressor(random_state={RANDOM_STATE})")
    print("=" * 70)

    model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        verbose=1,
    )

    t_start = time.time()
    model.fit(X_train, y_train)
    wall_clock = time.time() - t_start

    print(f"\n  ✓ Training completed in: {wall_clock:.2f} seconds ({wall_clock / 60:.2f} min)")

    # 3. Predict & Evaluate on Test Set
    print("\n" + "=" * 70)
    print("EVALUATING ON UNSEEN TEST SATELLITES")
    print("=" * 70)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("  Test Set Performance:")
    print(f"    - MAE:  {mae:.2f} days")
    print(f"    - RMSE: {rmse:.2f} days")

    # 4. Save Model Artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n  ✓ Model artifact saved: {MODEL_PATH}")

    # 5. Save Predicted vs Actual Plot
    print("\n" + "=" * 70)
    print("GENERATING PREDICTED VS ACTUAL PLOT")
    print("=" * 70)
    save_scatter_plot(
        y_test=y_test,
        y_pred=y_pred,
        model_name="Hist Gradient Boosting Regressor",
        output_paths=PLOT_PATHS,
        metrics={"MAE": mae, "RMSE": rmse},
    )

    print("\n" + "=" * 70)
    print("HISTOGRAM GRADIENT BOOSTING BASELINE COMPLETE")
    print(f"  Wall-Clock Time: {wall_clock:.2f}s | MAE: {mae:.2f} days | RMSE: {rmse:.2f} days")
    print("=" * 70)


if __name__ == "__main__":
    main()
