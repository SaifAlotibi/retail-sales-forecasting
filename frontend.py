import streamlit as st
import pandas as pd
import requests


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="centered"
)


# =========================================================
# Load Data
# =========================================================

@st.cache_data
def load_data():

    sales_data = pd.read_csv("train.csv")
    sales_data["date"] = pd.to_datetime(sales_data["date"])

    stores = pd.read_csv("stores.csv")

    holidays = pd.read_csv("holidays_events.csv")
    holidays["date"] = pd.to_datetime(holidays["date"])

    return sales_data, stores, holidays


sales_data, stores, holidays = load_data()


# =========================================================
# Create Features
# =========================================================

def create_features(store_nbr, family, target_date, onpromotion):

    history = sales_data[
        (sales_data["store_nbr"] == store_nbr) &
        (sales_data["family"] == family)
    ].copy()

    history = history.sort_values("date")

    target_date = pd.Timestamp(target_date)

    # Only use data before the prediction date
    history = history[history["date"] < target_date]

    # Need enough historical data
    if len(history) < 30:
        return None

    # -------------------------
    # Lag 1
    # -------------------------

    lag_1_date = target_date - pd.Timedelta(days=1)

    lag_1_data = history[
        history["date"] == lag_1_date
    ]

    if lag_1_data.empty:
        return None

    lag_1 = lag_1_data["sales"].iloc[0]

    # -------------------------
    # Lag 7
    # -------------------------

    lag_7_date = target_date - pd.Timedelta(days=7)

    lag_7_data = history[
        history["date"] == lag_7_date
    ]

    if lag_7_data.empty:
        return None

    lag_7 = lag_7_data["sales"].iloc[0]

    # -------------------------
    # Lag 30
    # -------------------------

    lag_30_date = target_date - pd.Timedelta(days=30)

    lag_30_data = history[
        history["date"] == lag_30_date
    ]

    if lag_30_data.empty:
        return None

    lag_30 = lag_30_data["sales"].iloc[0]

    # -------------------------
    # Rolling 7
    # -------------------------

    rolling_7_start = target_date - pd.Timedelta(days=7)

    rolling_7_data = history[
        (history["date"] >= rolling_7_start) &
        (history["date"] < target_date)
    ]

    if len(rolling_7_data) < 7:
        return None

    rolling_7 = rolling_7_data["sales"].mean()

    # -------------------------
    # Rolling 30
    # -------------------------

    rolling_30_start = target_date - pd.Timedelta(days=30)

    rolling_30_data = history[
        (history["date"] >= rolling_30_start) &
        (history["date"] < target_date)
    ]

    if len(rolling_30_data) < 30:
        return None

    rolling_30 = rolling_30_data["sales"].mean()

    return {
        "onpromotion": onpromotion,

        "year": target_date.year,
        "month": target_date.month,
        "day": target_date.day,
        "day_of_week": target_date.dayofweek,
        "week_of_year": target_date.isocalendar().week,

        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_30": lag_30,

        "rolling_7": rolling_7,
        "rolling_30": rolling_30
    }


# =========================================================
# Get Store Information
# =========================================================

def get_store_info(store_nbr):

    store = stores[
        stores["store_nbr"] == store_nbr
    ]

    if store.empty:
        return None

    store = store.iloc[0]

    return {
        "store_nbr": int(store["store_nbr"]),
        "city": store["city"],
        "state": store["state"],
        "type": store["type"],
        "cluster": int(store["cluster"])
    }


# =========================================================
# Get Holiday Information
# =========================================================

def get_holiday_info(target_date):

    target_date = pd.Timestamp(target_date)

    holiday = holidays[
        holidays["date"] == target_date
    ]

    if holiday.empty:

        return {
            "is_holiday": 0,
            "holiday_type": None,
            "locale": None,
            "transferred": False
        }

    return {
        "is_holiday": len(holiday),
        "holiday_type": holiday["type"].iloc[0],
        "locale": holiday["locale"].iloc[0],
        "transferred": bool(holiday["transferred"].max())
    }


# =========================================================
# Validate Forecast Date
# =========================================================

def validate_forecast_date(store_nbr, family, target_date):

    history = sales_data[
        (sales_data["store_nbr"] == store_nbr) &
        (sales_data["family"] == family)
    ]

    if history.empty:

        return False, "No historical data found for this store and product family."

    target_date = pd.Timestamp(target_date)

    latest_date = history["date"].max()
    earliest_date = history["date"].min()

    # Date must be after the beginning of the dataset
    if target_date <= earliest_date:

        return False, "Forecast date is too early."

    # Current model uses historical data
    if target_date > latest_date:

        return False, (
            f"Forecast date is outside the historical dataset. "
            f"Latest available date: {latest_date.date()}"
        )

    # Check whether enough historical data exists
    required_start = target_date - pd.Timedelta(days=30)

    available_history = history[
        (history["date"] >= required_start) &
        (history["date"] < target_date)
    ]

    if len(available_history) < 30:

        return False, "Not enough historical data for this date."

    return True, "Valid forecast date."


# =========================================================
# Build Final Prediction Input
# =========================================================

def build_prediction_input(
    store_nbr,
    family,
    target_date,
    onpromotion
):

    # Get time-series features
    features = create_features(
        store_nbr,
        family,
        target_date,
        onpromotion
    )

    if features is None:
        return None

    # Get store information
    store_info = get_store_info(store_nbr)

    if store_info is None:
        return None

    # Get holiday information
    holiday_info = get_holiday_info(target_date)

    # Combine everything
    input_data = {
        "store_nbr": store_info["store_nbr"],
        "family": family,

        "city": store_info["city"],
        "state": store_info["state"],
        "type": store_info["type"],
        "cluster": store_info["cluster"],

        "onpromotion": features["onpromotion"],

        "year": features["year"],
        "month": features["month"],
        "day": features["day"],
        "day_of_week": features["day_of_week"],
        "week_of_year": int(features["week_of_year"]),

        "lag_1": features["lag_1"],
        "lag_7": features["lag_7"],
        "lag_30": features["lag_30"],

        "rolling_7": features["rolling_7"],
        "rolling_30": features["rolling_30"],

        "is_holiday": holiday_info["is_holiday"],
        "holiday_type": holiday_info["holiday_type"],
        "locale": holiday_info["locale"],
        "transferred": holiday_info["transferred"]
    }

    return input_data


# =========================================================
# User Interface
# =========================================================

st.title("📈 Retail Sales Forecasting")

st.write(
    "Predict retail sales using historical sales patterns, "
    "store information, promotions, calendar features, and holidays."
)


st.divider()


# =========================================================
# Input Section
# =========================================================

st.subheader("Enter Prediction Details")


# Store selection
store_numbers = sorted(
    sales_data["store_nbr"].unique()
)

store_nbr = st.selectbox(
    "Store",
    store_numbers
)


# Product family selection
families = sorted(
    sales_data["family"].unique()
)

family = st.selectbox(
    "Product Family",
    families
)


# Forecast date
min_date = sales_data["date"].min().date()
max_date = sales_data["date"].max().date()

target_date = st.date_input(
    "Forecast Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)


# Promotion
onpromotion = st.number_input(
    "Items on Promotion",
    min_value=0,
    value=0,
    step=1
)


st.divider()


# =========================================================
# Prediction Button
# =========================================================

if st.button(
    "Predict Sales",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------
    # Validate date
    # -----------------------------------------

    valid, message = validate_forecast_date(
        store_nbr,
        family,
        target_date
    )

    if not valid:

        st.error(message)

    else:

        # -----------------------------------------
        # Build prediction input
        # -----------------------------------------

        input_data = build_prediction_input(
            store_nbr,
            family,
            target_date,
            onpromotion
        )

        if input_data is None:

            st.error(
                "Unable to create the required prediction features."
            )

        else:

            # -----------------------------------------
            # Send request to FastAPI
            # -----------------------------------------

            try:

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json=input_data
                )

                if response.status_code == 200:

                    result = response.json()

                    prediction = result["predicted_sales"]

                    # -----------------------------------------
                    # Display prediction
                    # -----------------------------------------

                    st.success("Prediction completed successfully!")

                    st.metric(
                        label="Predicted Sales",
                        value=f"{prediction:,.2f}"
                    )

                    st.write(
                        f"**Store:** {store_nbr}"
                    )

                    st.write(
                        f"**Product Family:** {family}"
                    )

                    st.write(
                        f"**Forecast Date:** {target_date}"
                    )

                    st.write(
                        f"**Items on Promotion:** {onpromotion}"
                    )

                    # -----------------------------------------
                    # Show generated features
                    # -----------------------------------------

                    with st.expander("View Generated Features"):

                        feature_df = pd.DataFrame(
                            [input_data]
                        ).T

                        feature_df.columns = ["Value"]

                        st.dataframe(
                            feature_df,
                            use_container_width=True
                        )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    st.write(response.text)

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server."
                )

                st.info(
                    "Make sure your FastAPI server is running with Uvicorn."
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {e}"
                )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Retail Sales Forecasting ML Project | "
    "XGBoost + FastAPI + Streamlit"
)