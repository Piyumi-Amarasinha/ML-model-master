**Sri Lanka Agri-Price Predictor**

## 📖 Project Overview
This machine learning project predicts the daily wholesale price of vegetables across different regions in Sri Lanka. By combining historical market prices with local climate data, the model provides short-term price forecasting to help farmers, traders, and stakeholders navigate agricultural market volatility. The pipeline supports **any of the 20 vegetable commodities** in the dataset (Carrot, Onion, Potato, Cabbage, Pumpkin, etc.) — not just carrots.

## ✨ Key Features
1. Multi-Commodity: Train and serve a model for any vegetable in the dataset, or train all of them at once.

2. Time-Series Forecasting: Utilizes a 7-day sliding window feature (Price_7_Days_Ago) to capture historical market trends, with the dashboard able to recursively forecast multiple days ahead.

3. Climate Integration: Factors in daily temperature, rainfall, and humidity to adjust price predictions based on weather anomalies.

4. Explainable AI (XAI): Implements SHAP (SHapley Additive exPlanations) to break down the "black box" and visually explain how weather and history impact the final price.

5. Interactive Dashboard: Features a live Streamlit web application allowing users to switch commodities, input custom market conditions, and forecast several days ahead.

## 🗄️ Dataset & Preprocessing
The model is trained on a compiled dataset spanning 2020–2025, containing wholesale vegetable prices and climate records across 25 districts in Sri Lanka, covering 20 distinct vegetable commodities.

Preprocessing Steps (`Preprocess/preprocessdata.py`):

1. Data Cleaning: Handled missing values and removed anomalous string characters from numeric columns (e.g., Crop Yield Impact Score).

2. Commodity Filtering: Filters the dataset down to a single vegetable commodity, selectable via `--commodity`.

3. Feature Engineering: Extracted temporal features (Year, Month, Day_of_Week) from raw Date strings to capture seasonal harvest cycles.

4. Lag Features: Shifted historical data to create the Price_7_Days_Ago predictor.

5. Encoding: Applied One-Hot Encoding to the categorical Region feature to ensure spatial data could be processed by the regression algorithm.

Usage:
```bash
# List the vegetable commodities available in the raw dataset
python Preprocess/preprocessdata.py --list-commodities

# Write a cleaned, ML-ready CSV for a specific commodity (optional - trainmodel.py
# calls this logic directly in-memory and doesn't require this step)
python Preprocess/preprocessdata.py --commodity Onion
```

## 🧠 Model Architecture & Evaluation
To capture the complex, non-linear relationships between weather and agricultural economics without relying on deep learning, this project utilizes XGBoost (Extreme Gradient Boosting).

The model was evaluated using a strict chronological time-series split (80% Train / 20% Test) to prevent data leakage.

Evaluation Metrics Used: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-Squared (R2) — each reported alongside a **naive baseline** (predicting that next week's price equals last week's price) so the model's improvement over a trivial forecast is explicit.

Training (`TrainModel/trainmodel.py`):
```bash
# Train Carrot with the default, fixed hyperparameters
python TrainModel/trainmodel.py --commodity Carrot

# Train Onion with hyperparameter tuning (RandomizedSearchCV + TimeSeriesSplit)
python TrainModel/trainmodel.py --commodity Onion --tune
```

Train every commodity in one pass (`TrainModel/train_all.py`):
```bash
python TrainModel/train_all.py
```
This writes a per-commodity model under `output/<commodity>/` and a cross-commodity comparison at `output/model_summary.csv`.

Each commodity's output folder (`output/<commodity>/`) contains:
- `model.pkl` — the trained XGBoost model
- `prediction_results.csv` — test-set features plus actual/predicted prices
- `metrics.json` — MAE/RMSE/R2 for the model and the naive baseline, plus tuned hyperparameters if `--tune` was used
- `shap_feature_importance.png`, `shap_summary_plot.png` — global SHAP plots
- `predictions_plot.png` — actual vs. predicted prices on the first 100 test samples

📊 Explainability (SHAP)
Transparency is critical in financial and agricultural forecasting. SHAP values were extracted from the trained XGBoost model to ensure the algorithm's behavior aligns with real-world domain knowledge.

Feature Importance: Historical prices proved to be the strongest baseline predictor, followed closely by rainfall and temperature.

Summary Analysis: The SHAP beeswarm plots confirm that extreme weather events (like unusually high rainfall) correctly drive the model to predict higher wholesale prices due to implied harvest disruption.

## 🚀 How to Run Locally
1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Train at least one commodity
```bash
python TrainModel/trainmodel.py --commodity Carrot
```

3. Run the Dashboard
```bash
streamlit run app/home.py
```
This starts a local server, and the dashboard opens in your default browser. Use the sidebar to switch between any commodity that has been trained. The "Market Inputs" panel takes the last 7 actual daily prices plus expected climate conditions, and forecasts forward up to 14 days — predictions beyond day 7 recursively reuse the model's own prior forecasts as the 7-day lag input.

## 🐳 Docker
```bash
docker build -f Dockerfile/Dockerfile -t agri-price .
docker run -p 8501:8501 agri-price
```
The image trains the default Carrot model at build time. Edit the `RUN python TrainModel/trainmodel.py --commodity Carrot` line in `Dockerfile/Dockerfile` (or swap it for `train_all.py`) to bake in other commodities.

## ✅ Running Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```
Tests run against small synthetic datasets (not the full 130k-row CSV) so they stay fast, and the same suite runs in CI on every push/PR (`.github/workflows/ci.yml`).
