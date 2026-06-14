from datetime import date, datetime
from typing import Optional, Literal
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from preprocess import get_merged_data, BAND_ORDER

app = FastAPI(title="Energy Weather API", version="1.0")

DATA = get_merged_data()
STORED_RECORDS = []


class DataPoint(BaseModel):
    model_config = {"populate_by_name": True}

    time: datetime
    total_load_actual: float = Field(alias="total load actual")
    total_load_forecast: float = Field(alias="total load forecast")
    price_actual: float = Field(alias="price actual")
    price_day_ahead: float = Field(alias="price day ahead")
    temp_c: float
    humidity: float
    wind_speed: float
    clouds_all: float
    generation_solar: float = Field(alias="generation solar")
    generation_wind_onshore: float = Field(alias="generation wind onshore")
    generation_fossil_gas: float = Field(alias="generation fossil gas")
    season: Literal["Winter", "Spring", "Summer", "Autumn"]
    time_of_day: Literal["Night", "Morning", "Afternoon", "Evening"]
    is_weekend: bool
    hour: int = Field(ge=0, le=23)
    month: int = Field(ge=1, le=12)
    renewable_share: float
    load_forecast_error: float


def _filter_dataframe(df, start_date, end_date, season, hour_val):
    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]
    if season is not None:
        df = df[df["season"] == season]
    if hour_val is not None:
        df = df[df["hour"] == hour_val]
    return df


@app.get("/data")
def get_data(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    season: Optional[str] = Query(None),
    hour: Optional[int] = Query(None, ge=0, le=23),
    fields: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(1000, ge=1, le=5000),
):
    filtered = _filter_dataframe(DATA, start_date, end_date, season, hour)

    if fields is not None:
        requested = [f.strip() for f in fields.split(",")]
        available = [c for c in requested if c in filtered.columns]
        if available:
            filtered = filtered[available]

    total = len(filtered)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    data_page = filtered.iloc[start_idx:end_idx].to_dict(orient="records")

    return {"total": total, "page": page, "limit": limit, "data": data_page}


@app.post("/data", status_code=201)
def create_data(point: DataPoint):
    record = {
        "time": point.time.isoformat(),
        "total load actual": point.total_load_actual,
        "total load forecast": point.total_load_forecast,
        "price actual": point.price_actual,
        "price day ahead": point.price_day_ahead,
        "temp_c": point.temp_c,
        "humidity": point.humidity,
        "wind_speed": point.wind_speed,
        "clouds_all": point.clouds_all,
        "generation solar": point.generation_solar,
        "generation wind onshore": point.generation_wind_onshore,
        "generation fossil gas": point.generation_fossil_gas,
        "season": point.season,
        "time_of_day": point.time_of_day,
        "is_weekend": point.is_weekend,
        "hour": point.hour,
        "month": point.month,
        "renewable_share": point.renewable_share,
        "load_forecast_error": point.load_forecast_error,
    }
    STORED_RECORDS.append(record)
    return {"message": "Record added", "stored_count": len(STORED_RECORDS), "record": record}


@app.get("/stored")
def get_stored():
    return {"total_stored": len(STORED_RECORDS), "records": STORED_RECORDS}


@app.get("/stats")
def get_stats(group_by: str = Query("season")):
    df = DATA.copy()
    if group_by == "season":
        groups = df.groupby("season", observed=True)
    elif group_by == "time_of_day":
        groups = df.groupby("time_of_day", observed=True)
    elif group_by == "temperature_band":
        df["temperature_band"] = pd.cut(
            df["temp_c"], bins=[-100, 15, 20, 25, 30, 100], labels=BAND_ORDER,
        )
        groups = df.groupby("temperature_band", observed=True)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown group_by: {group_by}")

    result = groups.agg(
        mean_load=("total load actual", "mean"),
        median_load=("total load actual", "median"),
        mean_temp=("temp_c", "mean"),
        mean_price=("price actual", "mean"),
        hours=("total load actual", "count"),
    ).round(2).reset_index().to_dict(orient="records")

    return {"group_by": group_by, "stats": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", reload=True)
