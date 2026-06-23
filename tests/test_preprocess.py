import pandas as pd
import pytest

from Preprocess.preprocessdata import clean_dataframe, list_commodities


def _synthetic_raw_df() -> pd.DataFrame:
    """Mimics the raw export schema: one row per (date, region, commodity) with a
    paired fruit reading, mangled column names, and one malformed numeric value."""
    regions = ["RegionA", "RegionB"]
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for region in regions:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Region": region,
                    "Temperature (°C)": 28.0 + i,
                    "Rainfall (mm)": 10.0 + i,
                    "Humidity (%)": 70.0 + i,
                    "Crop Yield Impact Score": "1.5s" if i == 0 else 1.0 + i * 0.1,
                    "fruit_Commodity": "Banana",
                    "fruit_Price per Unit (LKR/kg)": 80.0,
                    "vegitable_Commodity": "Carrot",
                    "vegitable_Price per Unit (LKR/kg)": 200.0 + i * 5,
                }
            )
            rows.append(
                {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Region": region,
                    "Temperature (°C)": 28.0 + i,
                    "Rainfall (mm)": 10.0 + i,
                    "Humidity (%)": 70.0 + i,
                    "Crop Yield Impact Score": 1.2,
                    "fruit_Commodity": "Banana",
                    "fruit_Price per Unit (LKR/kg)": 80.0,
                    "vegitable_Commodity": "Onion",
                    "vegitable_Price per Unit (LKR/kg)": 100.0 + i * 2,
                }
            )
    return pd.DataFrame(rows)


def test_clean_dataframe_filters_to_single_commodity():
    raw_df = _synthetic_raw_df()
    cleaned = clean_dataframe(raw_df.copy(), "Carrot")

    # 10 days per region; the first 7 are dropped to build the 7-day lag window,
    # leaving 3 valid rows per region (6 total across the 2 regions).
    assert len(cleaned) == 6
    assert "Price_7_Days_Ago" in cleaned.columns
    assert "Region_RegionB" in cleaned.columns
    assert "Region_RegionA" not in cleaned.columns  # dropped as the baseline category
    assert "Vegetable_Commodity" not in cleaned.columns
    assert "Fruit_Commodity" not in cleaned.columns
    assert not cleaned.isna().any().any()


def test_clean_dataframe_coerces_bad_impact_score():
    raw_df = _synthetic_raw_df()
    cleaned = clean_dataframe(raw_df.copy(), "Carrot")
    assert pd.api.types.is_numeric_dtype(cleaned["Crop_Yield_Impact_Score"])


def test_clean_dataframe_unknown_commodity_raises():
    raw_df = _synthetic_raw_df()
    with pytest.raises(SystemExit):
        clean_dataframe(raw_df.copy(), "DoesNotExist")


def test_list_commodities():
    raw_df = _synthetic_raw_df()
    assert list_commodities(raw_df.copy()) == ["Carrot", "Onion"]
