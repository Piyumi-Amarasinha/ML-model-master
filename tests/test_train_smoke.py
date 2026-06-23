import numpy as np
import pandas as pd

from TrainModel import trainmodel


def _synthetic_cleaned_df(n_rows: int = 30) -> pd.DataFrame:
    """A tiny stand-in for what Preprocess.preprocessdata.load_and_clean would
    return, so the training smoke test never touches the real 130k-row dataset."""
    dates = pd.date_range("2024-01-08", periods=n_rows, freq="D")
    rng = np.random.default_rng(42)
    base_price = 250 + np.cumsum(rng.normal(0, 2, size=n_rows))
    return pd.DataFrame(
        {
            "Date": dates,
            "Temperature_C": rng.uniform(25, 35, size=n_rows),
            "Rainfall_mm": rng.uniform(0, 50, size=n_rows),
            "Humidity_pct": rng.uniform(60, 90, size=n_rows),
            "Crop_Yield_Impact_Score": rng.uniform(0.5, 1.8, size=n_rows),
            "Vegetable_Price_LKR_kg": base_price,
            "Month": dates.month,
            "Day_of_Week": dates.dayofweek,
            "Year": dates.year,
            "Price_7_Days_Ago": np.roll(base_price, 7),
            "Region_RegionB": [i % 2 == 0 for i in range(n_rows)],
        }
    )


def test_train_for_commodity_smoke(tmp_path, monkeypatch):
    synthetic_df = _synthetic_cleaned_df()
    monkeypatch.setattr(trainmodel, "load_and_clean", lambda commodity: synthetic_df.copy())

    metrics = trainmodel.train_for_commodity(commodity="TestVeg", tune=False, base_dir=tmp_path)

    expected_keys = {
        "commodity",
        "trained_at",
        "n_train",
        "n_test",
        "tuned",
        "best_params",
        "model_mae",
        "model_rmse",
        "model_r2",
        "model_r",
        "baseline_mae",
        "baseline_rmse",
        "baseline_r2",
        "mae_improvement_pct",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["commodity"] == "TestVeg"
    assert metrics["tuned"] is False
    assert metrics["n_train"] + metrics["n_test"] == len(synthetic_df)

    output_dir = tmp_path / "output" / "TestVeg"
    assert (output_dir / "model.pkl").exists()
    assert (output_dir / "prediction_results.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "shap_feature_importance.png").exists()
    assert (output_dir / "shap_summary_plot.png").exists()
