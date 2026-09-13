import joblib
import pandas as pd

model = joblib.load("sales_forecasting_model.pkl")

def predict_sales(data): 
    df = pd.DataFrame([data])

    prediction = model.predict(df) 
    return prediction[0]

sample = {
    "store_nbr": 10,
    "family": "GROCERY I",
    "city": "Quito",
    "state": "Pichincha",
    "type": "A",
    "cluster": 6,

    "onpromotion": 10,

    "year": 2017,
    "month": 8,
    "day": 16,
    "day_of_week": 2,
    "week_of_year": 33,

    "lag_1": 500,
    "lag_7": 450,
    "lag_30": 480,

    "rolling_7": 470,
    "rolling_30": 465,

    "is_holiday": 0,
    "holiday_type": None,
    "locale": None,
    "transferred": False
}

prediction = predict_sales(sample)

print("Predicted sales:", prediction)