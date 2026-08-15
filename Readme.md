# FORESIGHT — Retail Demand Forecasting

FORESIGHT is a retail demand forecasting system that predicts how much of a product a store is expected to sell on future days.

The forecasting level is:

**Store + SKU + Day**

The project uses historical retail sales data, time-based demand features, and a LightGBM regression model to generate single-day and multi-day demand forecasts.

## Project Objective

> Given a store, a product (SKU), and historical sales data, how much of that product should we expect the store to sell on a future day?

FORESIGHT is designed to answer this question using historical demand patterns and store/product characteristics.

## Features

* Store-level demand forecasting
* SKU-level demand forecasting
* Daily demand prediction
* Historical demand-based lag features
* Rolling demand features
* Calendar features
* LightGBM regression
* Recursive multi-day forecasting
* Configurable forecast horizon from 1–30 days
* Streamlit web application

## Dataset

The processed daily demand dataset contains:

* **8,513,611 rows**
* **17 columns**
* Date range: **2022-01-01 to 2025-12-31**

The forecasting target is:

```text
daily_quantity
```

The forecasting data is organized at:

```text
Store + SKU + Day
```

## Feature Engineering

The final LightGBM model uses 13 features:

```text
store_id
sku_id
category
brand
store_city
store_type
year
month
day_of_week
lag_1_quantity
lag_7_quantity
rolling_7_quantity
is_weekend
```

### Lag Features

`lag_1_quantity` represents demand exactly one calendar day earlier.

`lag_7_quantity` represents demand exactly seven calendar days earlier.

If the required date does not exist, the corresponding lag value is set to `0`.

### Rolling Feature

`rolling_7_quantity` represents the mean of the previous 7 available demand observations while excluding the current forecast date.

## Model

The final selected model is **LightGBM Regressor**.

Configuration:

```python
lgb.LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
```

## Model Performance

The final model was evaluated using a chronological train/test split.

| Metric |  Score |
| ------ | -----: |
| MAE    | 1.0205 |
| RMSE   | 1.4441 |
| R²     | 0.6549 |

### Data Split

```text
Training data: 6,246,038 rows
Test data:     2,267,573 rows
```

The split was performed chronologically to preserve the time-series nature of the problem.

## Forecasting Architecture

FORESIGHT uses inference-time feature generation rather than simply selecting an existing historical test date.

The application follows this flow:

```text
Store
  ↓
SKU
  ↓
Forecast Date
  ↓
Historical Store + SKU Demand
  ↓
Build Forecast Features
  ↓
LightGBM
  ↓
Demand Prediction
```

For multi-day forecasting, FORESIGHT uses recursive forecasting:

```text
Day 1 prediction
      ↓
Added to working history
      ↓
Day 2 prediction
      ↓
Added to working history
      ↓
Day 3 prediction
      ↓
...
```

This allows the application to generate forecasts for multiple consecutive future days.

## Streamlit Application

The Streamlit application allows the user to select:

* Store
* SKU
* Forecast date
* Number of forecast days

The forecast horizon can be configured from **1 to 30 days**.

The application returns:

* Daily predicted demand
* Total forecasted demand

## Project Structure

```text
project_foresight/
│
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   ├── lightgbm_model.pkl
│   ├── lightgbm_model_info.pkl
│   └── model_comparison.csv
│
└── data/
    └── processed/
        └── daily_demand.csv
```

## Technologies Used

* Python
* Pandas
* NumPy
* LightGBM
* Scikit-learn
* Joblib
* Streamlit
* Git
* GitHub

## Running Locally

Clone the repository:

```bash
git clone <your-repository-url>
cd project_foresight
```

Create and activate the virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Saved Model Files

The trained model and its metadata are stored in:

```text
models/lightgbm_model.pkl
models/lightgbm_model_info.pkl
```

The model comparison results are stored in:

```text
models/model_comparison.csv
```

## Future Improvements

Potential future improvements include:

* Forecast confidence intervals
* Demand visualization
* Inventory optimization
* Reorder recommendations
* Promotion-aware forecasting
* Holiday and seasonal features
* Model monitoring
* Automated model retraining
* Cloud deployment and CI/CD

## Project Status

**Completed**

* Data understanding and EDA
* Data cleaning and preparation
* Data integration and feature engineering
* Model training and evaluation
* LightGBM model selection
* Future inference logic
* Inference validation
* Recursive multi-day forecasting
* Streamlit forecasting application
* Deployment preparation

## Author

**Aadi Jain**

FORESIGHT — Retail Demand Forecasting
