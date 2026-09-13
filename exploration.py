
# =========================================================
# Imports
# =========================================================

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# Load and Prepare Training Data
# =========================================================

df = pd.read_csv("train.csv")

# Convert date from string to datetime
df["date"] = pd.to_datetime(df["date"])

data = df.copy()


# =========================================================
# Feature Engineering
# =========================================================

# Extract useful calendar information from the date
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data["day_of_week"] = data["date"].dt.dayofweek
data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)


# Sort the data chronologically within each store and product family.
# This is necessary before creating lag and rolling features.
data = data.sort_values(
    ["store_nbr", "family", "date"]
)


# ---------------------------------------------------------
# Lag Features
# ---------------------------------------------------------

# Previous day's sales
data["lag_1"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .shift(1)
)

# Sales from the previous week
data["lag_7"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .shift(7)
)

# Sales from 30 days earlier
data["lag_30"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .shift(30)
)


# ---------------------------------------------------------
# Rolling Features
# ---------------------------------------------------------

# Average sales over the previous 7 days.
# shift(1) ensures the current day's sales are not used,
# preventing target leakage.
data["rolling_7"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

# Average sales over the previous 30 days
data["rolling_30"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .transform(
        lambda x: x.shift(1).rolling(30).mean()
    )
)


# Remove rows that do not have enough historical data
# to calculate the lag and rolling features.
data = data.dropna()


# =========================================================
# Add Store Information
# =========================================================

stores = pd.read_csv("stores.csv")

data = data.merge(
    stores,
    on="store_nbr",
    how="left"
)


# =========================================================
# Add Holiday Information
# =========================================================

holidays = pd.read_csv("holidays_events.csv")

holidays["date"] = pd.to_datetime(holidays["date"])

# Aggregate holidays by date before merging.
# This prevents duplicate sales rows when multiple holiday
# records exist for the same date.
holiday_features = (
    holidays.groupby("date")
    .agg(
        is_holiday=("type", "count"),
        holiday_type=("type", "first"),
        locale=("locale", "first"),
        transferred=("transferred", "max")
    )
    .reset_index()
)

data = data.merge(
    holiday_features,
    on="date",
    how="left"
)


# =========================================================
# Train / Test Split
# =========================================================

# Use a chronological split instead of a random split because
# this is a time-series forecasting problem.
split_date = data["date"].quantile(0.8)

train = data[data["date"] < split_date]
test = data[data["date"] >= split_date]


# =========================================================
# Define Features and Target
# =========================================================

features = [
    # Store information
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",

    # Promotion
    "onpromotion",

    # Calendar features
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",

    # Historical sales
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_7",
    "rolling_30",

    # Holiday information
    "is_holiday",
    "holiday_type",
    "locale",
    "transferred"
]

X_train = train[features]
y_train = train["sales"]

X_test = test[features]
y_test = test["sales"]


# =========================================================
# Define Categorical and Numerical Features
# =========================================================

categorical_features = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
    "holiday_type",
    "locale"
]

numerical_features = [
    "onpromotion",

    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",

    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_7",
    "rolling_30",

    "is_holiday",
    "transferred"
]


# =========================================================
# Preprocessing
# =========================================================

# One-hot encode categorical variables while keeping numerical
# features unchanged.
preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "num",
        "passthrough",
        numerical_features
    )
])


# =========================================================
# Build Machine Learning Pipeline
# =========================================================

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),

    (
        "model",
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
    )
])


# =========================================================
# Train Model
# =========================================================

model_pipeline.fit(
    X_train,
    y_train
)


# =========================================================
# Make Predictions
# =========================================================

predictions = model_pipeline.predict(X_test)


# =========================================================
# Model Evaluation
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

print("\nModel Performance")
print("-----------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)


# =========================================================
# Baseline Comparison
# =========================================================

# Use sales from the previous week as a simple baseline.
# This helps determine whether the ML model actually
# improves upon a basic forecasting approach.
baseline_predictions = X_test["lag_7"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

print("\nBaseline Performance")
print("--------------------")
print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)


# =========================================================
# Error Analysis
# =========================================================

errors = y_test.values - predictions

print("\nError Analysis")
print("--------------")
print("Mean Error:", errors.mean())
print("Median Error:", np.median(errors))
print("Max Error:", errors.max())
print("Min Error:", errors.min())


# =========================================================
# Feature Importance
# =========================================================

model = model_pipeline.named_steps["model"]
preprocessor_fitted = model_pipeline.named_steps["preprocessor"]

# Get the feature names after one-hot encoding
feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 20 Important Features")
print("-------------------------")
print(feature_importance.head(20))


# =========================================================
# Save Trained Model
# =========================================================

joblib.dump(
    model_pipeline,
    "sales_forecasting_model.pkl"
)

print("\nModel saved successfully!")


joblib.dump(
    model_pipeline, 
    "sales_forecasting_model.pkl")
print("Model saved successfully!")
