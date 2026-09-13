# Retail Sales Forecasting

Machine Learning project for forecasting retail sales using historical sales data, time-series features, store information, promotions, and holiday information.

The project uses **XGBoost** for prediction and provides a **FastAPI backend** and **Streamlit frontend** for making predictions through a simple web interface.

---

## Project Overview

The goal of this project is to predict retail sales for a specific:

* Store
* Product family
* Date
* Number of items on promotion

The model learns from historical sales patterns and additional information such as:

* Previous sales
* Rolling sales averages
* Store characteristics
* Product family
* Promotions
* Calendar features
* Holidays

---

## Dataset

The dataset comes from the Kaggle competition:

**Store Sales - Time Series Forecasting**

Dataset source:

https://www.kaggle.com/competitions/store-sales-time-series-forecasting

The original dataset contains daily sales information from multiple stores and product families in Ecuador.

Because the dataset is large, the raw CSV files are not included in this repository.

---

## Machine Learning Approach

### 1. Data Preparation

The data was loaded using Pandas and converted into a time-series format.

The dataset contains approximately 3 million training records.

The data includes:

* `store_nbr`
* `family`
* `date`
* `sales`
* `onpromotion`

---

### 2. Feature Engineering

Several features were created from the original data.

#### Calendar Features

* Year
* Month
* Day
* Day of week
* Week of year

#### Lag Features

Historical sales were used to create:

* `lag_1`
* `lag_7`
* `lag_30`

These represent sales from:

* Previous day
* Previous week
* Previous 30 days

#### Rolling Features

Rolling averages were also created:

* `rolling_7`
* `rolling_30`

These represent the average sales during the previous 7 and 30 days.

The rolling features were calculated using only previous observations to avoid data leakage.

---

### 3. Store Information

Store information was merged into the dataset.

The model uses:

* City
* State
* Store type
* Store cluster

---

### 4. Holiday Information

Holiday information was also incorporated into the model.

Features include:

* Holiday indicator
* Holiday type
* Locale
* Transferred holiday indicator

Holiday data was aggregated by date before merging to avoid duplicate rows.

---

## Model

The final model uses **XGBoost Regressor**.

Main configuration:

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
```

Categorical features were processed using `OneHotEncoder`.

The preprocessing and model were combined into a Scikit-learn `Pipeline`.

---

## Train/Test Split

Because this is a time-series problem, a random train/test split was not used.

Instead, the data was split chronologically:

* Earlier observations → Training data
* Later observations → Test data

This better represents how the model would be used in a real forecasting scenario.

---

## Model Evaluation

The final model achieved approximately:

| Metric |  Score |
| ------ | -----: |
| MAE    |  70.87 |
| RMSE   | 314.41 |

A simple baseline using `lag_7` achieved:

| Metric | Baseline |
| ------ | -------: |
| MAE    |    98.82 |
| RMSE   |   452.79 |

The XGBoost model improved the baseline MAE by approximately **28%**.

---

## Feature Importance

The most important features were primarily related to recent sales history.

The strongest features included:

1. `rolling_7`
2. `lag_1`
3. `lag_7`
4. Store information
5. Day of week
6. `rolling_30`
7. Holiday-related features
8. Promotion information

This shows that recent sales behavior is highly useful for forecasting future retail sales.

---

## Model Analysis

Several additional analyses were performed:

* Actual vs predicted sales
* Residual/error analysis
* Feature importance
* MAE by product family
* MAE by store
* Baseline comparison

The model showed very little overall prediction bias, although some high-sales observations produced significantly larger errors.

---

## Deployment

The trained model was integrated into a small prediction system.

### Backend

The backend was built using **FastAPI**.

The API provides:

```text
POST /predict
```

The endpoint receives the required prediction features and returns the predicted sales.

### Frontend

A **Streamlit** interface was created so users do not need to manually enter all engineered features.

The user only provides:

* Store
* Product family
* Forecast date
* Items on promotion

The application automatically calculates:

* Lag features
* Rolling averages
* Store information
* Holiday information
* Calendar features

The frontend then sends the generated data to the FastAPI backend.

---

## Project Structure

```text
retail-sales-forecasting/
│
├── train.py
├── predict.py
├── app.py
├── frontend.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

The trained model and raw dataset files are excluded from GitHub because of their size.

---

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd retail-sales-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download the Kaggle dataset and place the required CSV files in the project directory.

Required files include:

```text
train.csv
stores.csv
holidays_events.csv
```

### 4. Train the model

Run:

```bash
python train.py
```

This will train the model and create:

```text
sales_forecasting_model.pkl
```

### 5. Start the FastAPI backend

Run:

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Streamlit frontend

Open another terminal and run:

```bash
streamlit run frontend.py
```

The Streamlit application will open in your browser.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* FastAPI
* Streamlit
* Joblib

---

## Key Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* Regression
* Time-series forecasting
* Feature engineering
* Lag features
* Rolling statistics
* Categorical encoding
* Pipelines
* ColumnTransformer
* Data leakage prevention
* Chronological train/test splitting
* Model evaluation
* Baseline comparison
* Feature importance
* Model deployment
* REST APIs
* Streamlit applications

---

## Future Improvements

Possible improvements include:

* Multi-step forecasting
* More advanced time-series models
* Additional external features such as oil prices and transactions
* Hyperparameter optimization on a smaller representative dataset
* Model monitoring
* Cloud deployment
* Automated retraining
* Improved future-date forecasting

---

## Author

**Saif Al-Otaibi**

Computer Science Student

Interested in Machine Learning and AI.
