from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
ENERGY_PATH = DATA_DIR / "energy_dataset.csv"
WEATHER_PATH = DATA_DIR / "weather_features.csv"

COLORS = {
    "winter": "steelblue", "spring": "mediumseagreen",
    "summer": "darkorange", "autumn": "indianred",
    "load": "slateblue", "price": "tomato",
    "fossil": "mediumpurple", "renewable": "deepskyblue",
    "chart_bg": "lightgray", "grid": "dimgray",
}
SEASON_COLORS = {
    "Winter": "steelblue", "Spring": "mediumseagreen",
    "Summer": "darkorange", "Autumn": "indianred",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
TIME_ORDER = ["Night", "Morning", "Afternoon", "Evening"]
BAND_ORDER = ["< 15\u00b0C", "15\u201320\u00b0C", "20\u201325\u00b0C", "25\u201330\u00b0C", "\u2265 30\u00b0C"]


def get_merged_data() -> pd.DataFrame:
    energy_raw = pd.read_csv(ENERGY_PATH)
    weather_raw = pd.read_csv(WEATHER_PATH)

    energy_raw["time_utc"] = pd.to_datetime(energy_raw["time"], utc=True)
    weather_raw["time_utc"] = pd.to_datetime(weather_raw["dt_iso"], utc=True)

    empty_columns = energy_raw.columns[energy_raw.isna().all()].tolist()
    energy = energy_raw.drop(columns=empty_columns).copy()

    energy = energy.dropna(subset=["total load actual"]).copy()

    weather_raw["city_name"] = weather_raw["city_name"].str.strip()

    weather = weather_raw.drop_duplicates(subset=["time_utc", "city_name"]).copy()

    weather_numeric = [
        "temp", "temp_min", "temp_max", "pressure", "humidity",
        "wind_speed", "wind_deg", "rain_1h", "rain_3h", "snow_3h", "clouds_all",
    ]
    for col in weather_numeric:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    def mode_or_first(s):
        s_valid = s.dropna()
        if len(s_valid) == 0:
            return ""
        m = s_valid.mode()
        return m.iloc[0] if len(m) else s_valid.iloc[0]

    weather_hourly = (
        weather
        .groupby("time_utc", as_index=False)
        .agg({
            **{col: "mean" for col in weather_numeric},
            "weather_main": mode_or_first,
            "weather_description": mode_or_first,
            "city_name": "nunique",
        })
        .rename(columns={"city_name": "weather_city_count"})
    )

    data = (
        energy
        .merge(weather_hourly, on="time_utc", how="inner")
        .sort_values("time_utc")
        .reset_index(drop=True)
    )

    data = data.dropna().reset_index(drop=True)

    is_kelvin = data["temp"].max() > 200
    if is_kelvin:
        data["temp_c"] = data["temp"] - 273.15
        data["temp_min_c"] = data["temp_min"] - 273.15
        data["temp_max_c"] = data["temp_max"] - 273.15
    else:
        data["temp_c"] = data["temp"]
        data["temp_min_c"] = data["temp_min"]
        data["temp_max_c"] = data["temp_max"]

    data["date"] = data["time_utc"].dt.date
    data["year"] = data["time_utc"].dt.year
    data["month"] = data["time_utc"].dt.month
    data["day_of_week"] = data["time_utc"].dt.day_name()
    data["day_of_week_num"] = data["time_utc"].dt.dayofweek
    data["hour"] = data["time_utc"].dt.hour

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }

    def time_of_day(hour):
        if hour < 6:
            return "Night"
        if hour < 12:
            return "Morning"
        if hour < 18:
            return "Afternoon"
        return "Evening"

    data["season"] = pd.Categorical(
        data["month"].map(season_map), categories=SEASON_ORDER, ordered=True,
    )
    data["time_of_day"] = pd.Categorical(
        data["hour"].apply(time_of_day), categories=TIME_ORDER, ordered=True,
    )
    data["is_weekend"] = data["day_of_week_num"].ge(5).astype(int)
    data["is_peak_hour"] = data["hour"].isin([8, 9, 10, 11, 18, 19, 20, 21]).astype(int)

    data["load_forecast_error"] = data["total load actual"] - data["total load forecast"]
    data["absolute_forecast_error"] = data["load_forecast_error"].abs()

    renewable_cols = [
        "generation solar", "generation wind onshore",
        "generation hydro run-of-river and poundage",
        "generation hydro water reservoir", "generation other renewable",
    ]
    data["renewable_generation"] = data[renewable_cols].sum(axis=1)
    data["renewable_share"] = data["renewable_generation"] / data["total load actual"]

    return data


if __name__ == "__main__":
    df = get_merged_data()
    print(f"Clean merged data: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
