import numpy as np

def add_time_features(df):
    df = df.sort_values("DATE_TIME").reset_index(drop=True)

    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["DAY"] = df["DATE_TIME"].dt.day
    df["MONTH"] = df["DATE_TIME"].dt.month
    df["DAY_OF_WEEK"] = df["DATE_TIME"].dt.dayofweek

    # Ciklični prikaz
    df["HOUR_SIN"] = np.sin(2*np.pi*df["HOUR"]/24)
    df["HOUR_COS"] = np.cos(2*np.pi*df["HOUR"]/24)
    df["MONTH_SIN"] = np.sin(2*np.pi*df["MONTH"]/12)
    df["MONTH_COS"] = np.cos(2*np.pi*df["MONTH"]/12)

    return df.dropna(subset=["AC_POWER","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"])
