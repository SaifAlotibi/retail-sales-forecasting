import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# train.csv
df = pd.read_csv(r"train.csv")

df["date"] = pd.to_datetime(df["date"])

data = df.copy()

data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data["day_of_week"] = data["date"].dt.dayofweek
data["week_of_year"] = (
    data["date"].dt.isocalendar().week.astype(int)
)

data = data.sort_values(
    ["store_nbr", "family", "date"]
)
data["lag_1"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .shift(1)
)

data["lag_7"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .shift(7)
)

data["lag_30"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .shift(30)
    )

data["rolling_7"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

data["rolling_30"] = (
    data.groupby(["store_nbr", "family"])["sales"]
    .transform(lambda x:x.shift(1).rolling(30).mean())
)
data = data.dropna()

# Stores.csv
stores = pd.read_csv("stores.csv")

data = data.merge(
    stores,
    on="store_nbr",
    how="left"
)
# Holidays_events.csv 

holidays = pd.read_csv("holidays_events.csv")

holidays["date"] = pd.to_datetime(holidays["date"])

holiday_features = (
    holidays.groupby("date").agg(
        is_holiday = ("type", "count"), 
        holiday_type = ("type", "first"), 
        locale = ("locale", "first"), 
        transferred = ("transferred", "max")

    ).reset_index()
)


data = data.merge(
    holiday_features, 
    on="date", 
    how="left"
)

split_date = data["date"].quantile(0.8)

train = data[data["date"] < split_date]
test = data[data["date"] >= split_date]

features = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
    
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
    "holiday_type",
    "locale",
    "transferred"
]

X_train = train[features]
y_train = train["sales"]

X_test = test[features]
y_test = test["sales"]

categorical_features = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
    "holiday_type",
    "locale", 
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

model_pipeline = Pipeline([
    ("preprocessor", preprocessor), 
    ("model", 
     XGBRegressor(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth= 6, 
        random_state=42, 
    ))
])

model_pipeline.fit(X_train, y_train)

predictions = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

baseline_predictions = X_test["lag_7"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)

errors = y_test.values - predictions
print("Mean Error:", errors.mean()) 
print("Median Error:", np.median(errors))
print("Max Error:", errors.max())
print("Min Error:", errors.min())

model = model_pipeline.named_steps["model"]
preprocessor_fitted = model_pipeline.named_steps["preprocessor"]
features_name = preprocessor_fitted.get_feature_names_out()

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": features_name, 
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    "importance", 
    ascending=False
)

print(feature_importance.head(20))

joblib.dump(
    model_pipeline, 
    "sales_forecasting_model.pkl")
print("Model saved successfully!")