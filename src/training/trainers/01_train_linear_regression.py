"""
01_train_linear_regression.py

Trains and evaluates a baseline Linear Regression model (as a floor) for satellite decay prediction:
  - Trains exclusively on train-satellite observations.
  - Evaluates exclusively on test-satellite observations.
  - Measures wall-clock training time.
  - Computes test set MAE and RMSE.
  - Generates predicted-vs-actual scatter plot.
  - Saves the trained model artifact to models/linear_regression.joblib.

Run via:
  python src/training/trainers/01_train_linear_regression.py
"""

from pathlib import Path
import sys
import time
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Add src/training to import data_loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import load_train_test_data, save_scatter_plot

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "linear_regression.joblib"
PLOT_PATHS = [
    BASE_DIR / "data" / "plots" / "predicted_vs_actual_linear_regression.png",
    BASE_DIR / "reports" / "modeling" / "predicted_vs_actual_linear_regression.png",
]


def main():
    print("=" * 70)
    print("BASELINE MODEL 1: LINEAR REGRESSION (FLOOR)")
    print("=" * 70)

    # 1. Load data
    X_train, y_train, X_test, y_test, _ = load_train_test_data()

    # 2. Train model with wall-clock timing
    print("\n" + "=" * 70)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("=" * 70)

    model = LinearRegression()

    t_start = time.time()
    model.fit(X_train, y_train)
    wall_clock = time.time() - t_start

    print(f"  ✓ Training completed in: {wall_clock:.3f} seconds ({wall_clock / 60:.2f} min)")

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
        model_name="Linear Regression (Floor)",
        output_paths=PLOT_PATHS,
        metrics={"MAE": mae, "RMSE": rmse},
    )

    print("\n" + "=" * 70)
    print("LINEAR REGRESSION BASELINE COMPLETE")
    print(f"  Wall-Clock Time: {wall_clock:.2f}s | MAE: {mae:.2f} days | RMSE: {rmse:.2f} days")
    print("=" * 70)


if __name__ == "__main__":
    main()
