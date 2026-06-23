from __future__ import annotations

import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st


st.set_page_config(page_title="Agri-Price Predictor", layout="wide", initial_sidebar_state="expanded")

PRICE_COLUMN = "Price (LKR/kg)"


def _set_first_existing(input_row: dict, candidates: list[str], value) -> None:
    for candidate in candidates:
        if candidate in input_row:
            input_row[candidate] = value
            return
    raise KeyError(
        f"None of the expected feature columns exist: {candidates}. "
        "Re-train the model or update the app feature mapping."
    )


def build_feature_row(
    expected_columns: list[str],
    *,
    temp: float,
    rainfall: float,
    humidity: float,
    impact_score: float,
    lag_price: float,
    year: int,
    month: int,
    day_of_week: int,
    region_col: str | None,
) -> dict:
    row = dict.fromkeys(expected_columns, 0)
    _set_first_existing(row, ["Temperature_C", "Temperature (°C)"], temp)
    _set_first_existing(row, ["Rainfall_mm"], rainfall)
    _set_first_existing(row, ["Humidity_pct"], humidity)
    _set_first_existing(row, ["Crop_Yield_Impact_Score"], impact_score)
    _set_first_existing(row, ["Price_7_Days_Ago"], lag_price)
    _set_first_existing(row, ["Year"], year)
    _set_first_existing(row, ["Month"], month)
    _set_first_existing(row, ["Day_of_Week"], day_of_week)
    if region_col and region_col in row:
        row[region_col] = 1
    return row


base_dir = Path(__file__).resolve().parents[1]
output_root = base_dir / "output"
available_commodities = sorted(
    p.name for p in output_root.glob("*") if p.is_dir() and (p / "model.pkl").exists()
) if output_root.exists() else []

if not available_commodities:
    st.error(
        "No trained models found in the 'output' folder.\n\n"
        "Run `python TrainModel/trainmodel.py --commodity Carrot` "
        "(or `python TrainModel/train_all.py` for every commodity) first."
    )
    st.stop()

default_index = available_commodities.index("Carrot") if "Carrot" in available_commodities else 0

st.sidebar.title("ℹ️ About this Application")
selected_commodity = st.sidebar.selectbox("Commodity", available_commodities, index=default_index)
st.sidebar.info(
    f"**Live {selected_commodity} Price Predictor**\n\n"
    f"This machine learning dashboard predicts the daily wholesale price of {selected_commodity.lower()} "
    "across 25 districts in Sri Lanka, using an **XGBoost Regressor** trained on historical climate and "
    "market data from 2020 to 2025.\n\n"
    "**Key Predictive Features:**\n"
    "- 7-Day Historical Price Lag\n"
    "- Weather anomalies (Rainfall, Temp, Humidity)\n"
    "- Regional Market Differences\n\n"
    "*Developed as a Machine Learning academic project focusing on local dataset compilation, "
    "advanced algorithm selection, and Explainable AI.*"
)

st.title(f"Live {selected_commodity} Price Predictor")
st.write(
    "Enter the last 7 days of actual prices plus expected climate conditions to forecast "
    f"upcoming wholesale {selected_commodity.lower()} prices."
)

commodity_dir = output_root / selected_commodity
model = joblib.load(commodity_dir / "model.pkl")
df_sample = pd.read_csv(commodity_dir / "prediction_results.csv")

expected_columns = [col for col in df_sample.columns if col not in ["Actual_Price", "Predicted_Price"]]
region_cols = [col for col in expected_columns if col.startswith("Region_")]
available_regions = ["Ampara"] + [col.replace("Region_", "") for col in region_cols]

st.subheader("Market Inputs")
col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input("Forecast Start Date", datetime.date.today())
    selected_region = st.selectbox("Select Region", sorted(available_regions))
    forecast_horizon = st.slider("Forecast Horizon (days)", min_value=1, max_value=14, value=7)

    st.caption("Last 7 days of actual prices (oldest to newest):")
    history_dates = [selected_date - datetime.timedelta(days=n) for n in range(7, 0, -1)]
    default_history = pd.DataFrame({"Date": history_dates, PRICE_COLUMN: [250.0] * 7})
    edited_history = st.data_editor(
        default_history,
        hide_index=True,
        disabled=["Date"],
        column_config={"Date": st.column_config.DateColumn("Date")},
        key="history_editor",
        use_container_width=True,
    )

with col2:
    st.caption("Expected climate conditions, held constant across the forecast horizon:")
    temp = st.number_input("Average Temperature (°C)", value=30.0, step=0.5, min_value=15.0, max_value=40.0)
    rainfall = st.number_input("Rainfall (mm)", value=15.0, step=1.0, min_value=0.0, max_value=500.0)
    humidity = st.number_input("Humidity (%)", value=75.0, step=1.0, min_value=0.0, max_value=100.0)
    impact_score = st.number_input("Crop Yield Impact Score (0.0 - 2.0)", value=1.50, step=0.1, min_value=0.0, max_value=2.0)

st.markdown("---")
if st.button("🔮 Predict & Forecast", type="primary"):
    recent_prices = edited_history[PRICE_COLUMN].tolist()
    region_col_name = f"Region_{selected_region}" if selected_region != "Ampara" else None

    forecast_prices: list[float] = []
    first_input_df: pd.DataFrame | None = None

    for i in range(forecast_horizon):
        forecast_date = selected_date + datetime.timedelta(days=i)
        # Day i's "price 7 days ago" is one of the user-supplied recent prices for the
        # first 7 forecast days; beyond that it chains onto the model's own prior predictions.
        lag_price = recent_prices[i] if i < 7 else forecast_prices[i - 7]

        row = build_feature_row(
            expected_columns,
            temp=temp,
            rainfall=rainfall,
            humidity=humidity,
            impact_score=impact_score,
            lag_price=lag_price,
            year=forecast_date.year,
            month=forecast_date.month,
            day_of_week=forecast_date.weekday(),
            region_col=region_col_name,
        )
        input_df = pd.DataFrame([row])[expected_columns]
        if i == 0:
            first_input_df = input_df

        predicted_price = float(model.predict(input_df)[0])
        forecast_prices.append(predicted_price)

    st.success(f"### 📈 Predicted Wholesale Price on {selected_date}: Rs. {forecast_prices[0]:.2f} per kg")

    st.markdown("---")
    st.subheader("📋 Input Summary")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    sum_col1.metric("Forecast Start Date", str(selected_date))
    sum_col1.metric("Region", selected_region)
    sum_col2.metric("Forecast Horizon", f"{forecast_horizon} days")
    sum_col2.metric("Temperature", f"{temp} °C")
    sum_col3.metric("Rainfall", f"{rainfall} mm")
    sum_col3.metric("Humidity", f"{humidity} %")

    st.markdown("---")
    st.subheader("📈 Price Forecast")
    if forecast_horizon > 7:
        st.caption(
            "Note: forecasts beyond day 7 reuse the model's own earlier predictions as the "
            "7-day lag input, so uncertainty compounds the further out you forecast."
        )
    forecast_dates = [selected_date + datetime.timedelta(days=i) for i in range(forecast_horizon)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history_dates, recent_prices, label="Actual (last 7 days)", color="blue", marker="o")
    ax.plot(forecast_dates, forecast_prices, label="Forecast", color="red", linestyle="dashed", marker="x")
    ax.set_xlabel("Date")
    ax.set_ylabel(PRICE_COLUMN)
    ax.set_title(f"{selected_commodity} Price: Recent Actuals + Forecast")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("📊 Prediction Explanation (SHAP Waterfall Plot)")
    st.write(
        "This waterfall plot breaks down how your inputs pushed the first forecast day's "
        "prediction up or down from the baseline."
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(first_input_df)

    fig = plt.figure(figsize=(4, 3))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("🌐 Global Model Analysis")
    st.write("These charts explain what the model learned from the entire dataset.")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.write("**Feature Importance (Bar Plot)**")
        image_path = commodity_dir / "shap_feature_importance.png"
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"Could not find {image_path.name} for {selected_commodity}.")

    with img_col2:
        st.write("**SHAP Beeswarm Summary**")
        image_path = commodity_dir / "shap_summary_plot.png"
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"Could not find {image_path.name} for {selected_commodity}.")
