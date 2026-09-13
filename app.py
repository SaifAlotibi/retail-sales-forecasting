import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("sales_forecasting_model.pkl")

app = FastAPI(
    title="Retail Sales Forecasting API",
    description="API for predicting retail sales",
    version="1.0"
)


class SalesInput(BaseModel):
    store_nbr: int
    family: str
    city: str
    state: str
    type: str
    cluster: int

    onpromotion: int

    year: int
    month: int
    day: int
    day_of_week: int
    week_of_year: int

    lag_1: float
    lag_7: float
    lag_30: float

    rolling_7: float
    rolling_30: float

    is_holiday: int
    holiday_type: str | None = None
    locale: str | None = None
    transferred: bool = False


@app.get("/")
def home():
    return {"message": "Retail Sales Forecasting API is running"}


@app.post("/predict")
def predict_sales(data: SalesInput):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_sales": float(prediction)
    }