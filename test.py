import pandas as pd

holidays = pd.read_csv("holidays_events.csv")

print(holidays["date"].duplicated().sum())
print(holidays[holidays["date"].duplicated(keep=False)].sort_values("date").head(20))

holiday_features = (
    holidays.groupby("date").agg(
        is_holiday = ("type", "count"), 
        holiday_type = ("type", "first"), 
        locale = ("locale", "first"), 
        transferred = ("transferred", "max")

    ).reset_index()
)

print(holiday_features.shape)
print(holiday_features.head())
print("Duplicated: ", holiday_features["date"].duplicated().sum())