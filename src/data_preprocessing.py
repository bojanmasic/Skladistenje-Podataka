import pandas as pd

def load_and_merge(gen_path, weather_path):
    gen = pd.read_csv(gen_path)
    weather = pd.read_csv(weather_path)

    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], errors="coerce")
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")

    gen = gen.dropna(subset=["DATE_TIME"])
    weather = weather.dropna(subset=["DATE_TIME"])

    gen_agg = gen.groupby(["PLANT_ID","DATE_TIME"], as_index=False).agg({
        "AC_POWER":"sum",
        "DC_POWER":"sum"
    })

    weather_agg = weather.groupby(["PLANT_ID","DATE_TIME"], as_index=False).agg({
        "AMBIENT_TEMPERATURE":"mean",
        "MODULE_TEMPERATURE":"mean",
        "IRRADIATION":"mean"
    })

    df = pd.merge(gen_agg, weather_agg, on=["PLANT_ID","DATE_TIME"], how="inner")
    return df
