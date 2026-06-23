from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Preprocess.preprocessdata import load_and_clean  # noqa: E402

DEFAULT_COMMODITY = "Carrot"

TUNE_PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}


def train_for_commodity(commodity: str = DEFAULT_COMMODITY, tune: bool = False, base_dir: Path = BASE_DIR) -> dict:
    print(f"Loading and cleaning dataset for commodity='{commodity}'...")
    df = load_and_clean(commodity)

    # 1. Ensure the data is strictly chronological for time-series predicting
    df = df.sort_values(by=['Year', 'Month', 'Date'])

    # XGBoost requires all data to be numbers.
    # We convert our One-Hot Encoded True/False columns into 1s and 0s.
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Drop the 'Date' string column since the model only reads numbers
    # (We already extracted Month, Day_of_Week, and Year earlier)
    df = df.drop(columns=['Date'])

    # 2. Define Features (X) and Target (y)
    X = df.drop(columns=['Vegetable_Price_LKR_kg'])
    y = df['Vegetable_Price_LKR_kg']

    # 3. Time-Series Train/Test Split (80% Train, 20% Test)
    # We CANNOT use a random split here. If we did, the model would peek into the future to predict the past.
    # We must train on the oldest 80% of data, and test on the newest 20%.
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training on {len(X_train)} historical records...")
    print(f"Testing on {len(X_test)} future records...")

    best_params = None
    if tune:
        print("Running RandomizedSearchCV with TimeSeriesSplit (this may take a while)...")
        search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(random_state=42),
            param_distributions=TUNE_PARAM_DISTRIBUTIONS,
            n_iter=20,
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=5),
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        print(f"Best params: {best_params}")
    else:
        # 4. Initialize and Train the XGBoost Model (Hyperparameters)
        # n_estimators: Number of sequential trees built.
        # learning_rate: Controls how aggressively each tree corrects the last one to prevent overfitting.
        # max_depth: How deep each decision tree is allowed to go.
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        model.fit(X_train, y_train)

    # 5. Evaluate the Model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    r = np.corrcoef(y_test, predictions)[0, 1]

    # Naive baseline for comparison: "next week's price = last week's price",
    # i.e. just reusing the Price_7_Days_Ago feature itself as the prediction.
    baseline_predictions = X_test['Price_7_Days_Ago']
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
    baseline_r2 = r2_score(y_test, baseline_predictions)
    mae_improvement_pct = (baseline_mae - mae) / baseline_mae * 100 if baseline_mae else float("nan")

    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(f"Mean Absolute Error (MAE): Rs. {mae:.2f}  (naive baseline: Rs. {baseline_mae:.2f}, {mae_improvement_pct:+.1f}% vs baseline)")
    print(f"Root Mean Squared Error (RMSE): Rs. {rmse:.2f}  (naive baseline: Rs. {baseline_rmse:.2f})")
    print(f"R (Correlation): {r:.4f}")
    print(f"R^2 (R-squared): {r2:.4f}  (naive baseline: {baseline_r2:.4f})")

    # 6. Save plots, model, and metrics under output/<commodity>/
    output_dir = base_dir / "output" / commodity
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    # Plotting just the first 100 predictions so the graph is readable
    plt.plot(y_test.values[:100], label='Actual Price', color='blue', marker='o')
    plt.plot(predictions[:100], label='Predicted Price', color='red', linestyle='dashed', marker='x')
    plt.title(f'XGBoost Predictions vs Actual {commodity} Prices (First 100 Test Samples)')
    plt.xlabel('Time (Test Samples)')
    plt.ylabel('Price (LKR/kg)')
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "predictions_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved predictions plot to '{plot_path}'")

    print("\nCalculating SHAP values (this might take a few seconds)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance (Bar) - {commodity}')
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'SHAP Summary Plot (Impact on {commodity} Price)')
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_plot.png")
    plt.close()
    print("Success! SHAP plots saved.")

    # 7. Save the trained model and test predictions for the frontend
    joblib.dump(model, output_dir / "model.pkl")

    results_df = X_test.copy()
    results_df['Actual_Price'] = y_test
    results_df['Predicted_Price'] = predictions
    results_df.to_csv(output_dir / "prediction_results.csv", index=False)

    metrics = {
        "commodity": commodity,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "tuned": tune,
        "best_params": best_params,
        "model_mae": mae,
        "model_rmse": rmse,
        "model_r2": r2,
        "model_r": r,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "mae_improvement_pct": mae_improvement_pct,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSuccess! Model, data, and metrics saved to '{output_dir}'.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost price model for a single vegetable commodity.")
    parser.add_argument("--commodity", "-c", default=DEFAULT_COMMODITY, help=f"Vegetable commodity to train on (default: {DEFAULT_COMMODITY}).")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV with time-series cross-validation instead of using fixed hyperparameters.",
    )
    args = parser.parse_args()
    train_for_commodity(commodity=args.commodity, tune=args.tune)


if __name__ == "__main__":
    main()
