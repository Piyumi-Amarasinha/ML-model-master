from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Preprocess.preprocessdata import _read_csv_with_encoding_fallback, _resolve_input_csv_path, list_commodities  # noqa: E402
from TrainModel.trainmodel import train_for_commodity  # noqa: E402


def main() -> None:
    input_path = _resolve_input_csv_path(None)
    raw_df = _read_csv_with_encoding_fallback(input_path, None, None)
    commodities = list_commodities(raw_df)

    print(f"Training models for {len(commodities)} commodities: {', '.join(commodities)}\n")

    summary_rows = []
    for commodity in commodities:
        print(f"\n===== {commodity} =====")
        metrics = train_for_commodity(commodity=commodity, tune=False, base_dir=BASE_DIR)
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)[
        ["commodity", "n_train", "n_test", "model_mae", "model_rmse", "model_r2", "baseline_mae", "mae_improvement_pct"]
    ]
    summary_path = BASE_DIR / "output" / "model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary across all commodities to '{summary_path}'.")


if __name__ == "__main__":
    main()
