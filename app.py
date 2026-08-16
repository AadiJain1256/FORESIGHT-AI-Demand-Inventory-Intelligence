
from datetime import date, timedelta

import streamlit as st
import joblib
import pandas as pd
import pyarrow.dataset as ds


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Foresight",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "models/lightgbm_model.pkl"
)

model.info = joblib.load(
    "models/lightgbm_model_info.pkl"
)

lgb_features = model.info["features"]
categorical_features = model.info["categorical_features"]


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Foresight")
st.subheader("Retail Demand Forecasting")

st.write(
    "AI-powered demand forecasting using LightGBM."
)

st.write(
    "Model features loaded:",
    len(lgb_features)
)

st.success(
    "LightGBM model loaded successfully"
)


# =========================================================
# LOAD STORE OPTIONS
# =========================================================

@st.cache_data
def load_store_options():

    dataset = ds.dataset(
        "data/processed/daily_demand.parquet",
        format="parquet"
    )

    store_values = set()

    scanner = dataset.scanner(
        columns=["store_id"],
        batch_size=100_000
    )

    for batch in scanner.to_batches():

        store_values.update(
            batch.column("store_id").to_pylist()
        )

    return sorted(store_values)


# =========================================================
# LOAD SKU OPTIONS FOR SELECTED STORE
# =========================================================

@st.cache_data
def load_sku_options(selected_store):

    dataset = ds.dataset(
        "data/processed/daily_demand.parquet",
        format="parquet"
    )

    sku_values = set()

    scanner = dataset.scanner(
        columns=["store_id", "sku_id"],
        filter=(
            ds.field("store_id") == selected_store
        ),
        batch_size=100_000
    )

    for batch in scanner.to_batches():

        sku_values.update(
            batch.column("sku_id").to_pylist()
        )

    return sorted(sku_values)


# =========================================================
# LOAD ONLY SELECTED STORE + SKU DATA
# =========================================================

@st.cache_data
def load_forecast_data(
    selected_store,
    selected_sku
):

    return pd.read_parquet(
        "data/processed/daily_demand.parquet",
        columns=lgb_features + [
            "date",
            "daily_quantity"
        ],
        filters=[
            ("store_id", "==", selected_store),
            ("sku_id", "==", selected_sku)
        ]
    )


# =========================================================
# BUILD FORECAST FEATURES
# =========================================================

def build_forecast_features(
    historical_data,
    selected_store,
    selected_sku,
    selected_date
):

    # -----------------------------------------------------
    # 1. Select the requested Store + SKU
    # -----------------------------------------------------

    historical_data = historical_data[
        (historical_data["store_id"] == selected_store)
        &
        (historical_data["sku_id"] == selected_sku)
    ].copy()


    # -----------------------------------------------------
    # 2. Convert date to datetime
    # -----------------------------------------------------

    historical_data["date"] = pd.to_datetime(
        historical_data["date"]
    )


    # -----------------------------------------------------
    # 3. Sort chronologically
    # -----------------------------------------------------

    historical_data = historical_data.sort_values(
        "date"
    )


    # -----------------------------------------------------
    # 4. Convert selected date to Timestamp
    # -----------------------------------------------------

    forecast_date = pd.Timestamp(
        selected_date
    )


    # -----------------------------------------------------
    # 5. Keep only observations before forecast date
    # -----------------------------------------------------

    historical_before_forecast = historical_data[
        historical_data["date"] < forecast_date
    ].copy()


    # -----------------------------------------------------
    # 6. Calculate lag_1_quantity
    #
    # Demand must exist exactly one calendar day earlier.
    # Otherwise the value is 0.
    # -----------------------------------------------------

    yesterday = (
        forecast_date -
        pd.Timedelta(days=1)
    )

    lag_1_row = historical_before_forecast[
        historical_before_forecast["date"] == yesterday
    ]

    lag_1_quantity = (
        lag_1_row["daily_quantity"].iloc[0]
        if not lag_1_row.empty
        else 0
    )


    # -----------------------------------------------------
    # 7. Calculate lag_7_quantity
    #
    # Demand must exist exactly seven calendar days earlier.
    # Otherwise the value is 0.
    # -----------------------------------------------------

    seven_days_ago = (
        forecast_date -
        pd.Timedelta(days=7)
    )

    lag_7_row = historical_before_forecast[
        historical_before_forecast["date"] == seven_days_ago
    ]

    lag_7_quantity = (
        lag_7_row["daily_quantity"].iloc[0]
        if not lag_7_row.empty
        else 0
    )


    # -----------------------------------------------------
    # 8. Calculate rolling_7_quantity
    #
    # Average of the previous 7 available observations.
    # The forecast date itself is excluded.
    # -----------------------------------------------------

    rolling_7_quantity = (
        historical_before_forecast[
            "daily_quantity"
        ]
        .tail(7)
        .mean()
    )


    # -----------------------------------------------------
    # 9. Calendar features
    # -----------------------------------------------------

    year = forecast_date.year

    month = forecast_date.month

    day_of_week = forecast_date.dayofweek

    is_weekend = (
        1
        if day_of_week >= 5
        else 0
    )


    # -----------------------------------------------------
    # 10. Store/SKU attributes
    # -----------------------------------------------------

    if not historical_before_forecast.empty:

        latest_row = historical_before_forecast.iloc[-1]

    else:

        latest_row = historical_data.iloc[0]


    category = latest_row["category"]

    brand = latest_row["brand"]

    store_city = latest_row["store_city"]

    store_type = latest_row["store_type"]


    # -----------------------------------------------------
    # 11. Build exact 13-feature model input
    # -----------------------------------------------------

    future_row = pd.DataFrame([{

        "store_id": selected_store,

        "sku_id": selected_sku,

        "category": category,

        "brand": brand,

        "store_city": store_city,

        "store_type": store_type,

        "year": year,

        "month": month,

        "day_of_week": day_of_week,

        "lag_1_quantity": lag_1_quantity,

        "lag_7_quantity": lag_7_quantity,

        "rolling_7_quantity": rolling_7_quantity,

        "is_weekend": is_weekend

    }])


    return future_row


# =========================================================
# RECURSIVE MULTI-DAY FORECAST
# =========================================================

def recursive_forecast(
    historical_data,
    selected_store,
    selected_sku,
    start_date,
    forecast_days
):

    # -----------------------------------------------------
    # 1. Create working copy
    # -----------------------------------------------------

    working_data = historical_data.copy()


    # -----------------------------------------------------
    # 2. Convert dates to datetime
    # -----------------------------------------------------

    working_data["date"] = pd.to_datetime(
        working_data["date"]
    )


    # -----------------------------------------------------
    # 3. Sort chronologically
    # -----------------------------------------------------

    working_data = working_data.sort_values(
        "date"
    )


    # -----------------------------------------------------
    # 4. Store forecast results
    # -----------------------------------------------------

    forecast_results = []


    # -----------------------------------------------------
    # 5. Forecast one day at a time
    # -----------------------------------------------------

    for i in range(forecast_days):

        forecast_date = (
            pd.Timestamp(start_date)
            +
            pd.Timedelta(days=i)
        )


        # -------------------------------------------------
        # Build inference features
        # -------------------------------------------------

        future_row = build_forecast_features(
            working_data,
            selected_store,
            selected_sku,
            forecast_date
        )


        # -------------------------------------------------
        # Match LightGBM categorical categories
        # -------------------------------------------------

        for i, col in enumerate(
            categorical_features
        ):

            future_row[col] = pd.Categorical(
                future_row[col],
                categories=(
                    model.booster_
                    .pandas_categorical[i]
                )
            )


        # -------------------------------------------------
        # Predict demand
        # -------------------------------------------------

        prediction = model.predict(
            future_row[lgb_features]
        )[0]


        # -------------------------------------------------
        # Store prediction
        # -------------------------------------------------

        forecast_results.append({

            "date": forecast_date,

            "store_id": selected_store,

            "sku_id": selected_sku,

            "predicted_quantity": prediction

        })


        # -------------------------------------------------
        # Add prediction to working history
        #
        # Today's prediction becomes historical demand
        # for the next forecast day.
        # -------------------------------------------------

        predicted_row = future_row.copy()

        predicted_row["date"] = forecast_date

        predicted_row["daily_quantity"] = prediction


        working_data = pd.concat(
            [
                working_data,
                predicted_row
            ],
            ignore_index=True
        )


        # Keep chronological order

        working_data = working_data.sort_values(
            "date"
        )


    # -----------------------------------------------------
    # 6. Convert results to DataFrame
    # -----------------------------------------------------

    forecast_results = pd.DataFrame(
        forecast_results
    )


    return forecast_results


# =========================================================
# DEMAND FORECAST SECTION
# =========================================================

st.header("Demand Forecast")


# =========================================================
# STORE SELECTION
# =========================================================

store_options = load_store_options()

selected_store = st.selectbox(
    "Select store",
    store_options
)


# =========================================================
# SKU SELECTION
# =========================================================

sku_options = load_sku_options(
    selected_store
)

selected_sku = st.selectbox(
    "Select SKU",
    sku_options
)


# =========================================================
# FORECAST DATE
# =========================================================
# Historical and future dates are allowed intentionally.

selected_date = st.date_input(
    "Select Forecast Date",
    value=date.today() + timedelta(days=1),
    min_value=None
)


# =========================================================
# FORECAST HORIZON
# =========================================================

forecast_days = st.number_input(
    "Number of Forecast Days",
    min_value=1,
    max_value=30,
    value=7,
    step=1
)


# =========================================================
# FORECAST BUTTON
# =========================================================

if st.button("Forecast Demand"):

    # -----------------------------------------------------
    # Load only the selected Store + SKU history
    # -----------------------------------------------------

    daily_demand_app = load_forecast_data(
        selected_store,
        selected_sku
    )


    # -----------------------------------------------------
    # Run recursive forecast
    # -----------------------------------------------------

    recursive_results = recursive_forecast(
        daily_demand_app,
        selected_store,
        selected_sku,
        selected_date,
        int(forecast_days)
    )


    # -----------------------------------------------------
    # Forecast result section
    # -----------------------------------------------------

    st.subheader("Demand Forecast")


    # -----------------------------------------------------
    # Total forecasted demand
    # -----------------------------------------------------

    total_demand = (
        recursive_results[
            "predicted_quantity"
        ].sum()
    )

    st.metric(
        "Total Forecasted Demand",
        f"{total_demand:.2f} units"
    )


    # -----------------------------------------------------
    # Daily forecast table
    # -----------------------------------------------------

    st.dataframe(
        recursive_results,
        use_container_width=True,
        hide_index=True
    )