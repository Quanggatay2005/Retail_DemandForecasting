

import json
import os
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# ── Load model info được save từ notebook 02 ────────────────────────────────
MODEL_INFO_PATH = "src/models/model_info.json"

if not os.path.exists(MODEL_INFO_PATH):
    raise FileNotFoundError(
        "Không tìm thấy models/model_info.json. "
        "Hãy chạy notebook 02_training_evaluation.ipynb trước."
    )

with open(MODEL_INFO_PATH) as f:
    MODEL_INFO = json.load(f)

FEATURE_COLS = MODEL_INFO["feature_cols"]
TARGET_COL   = MODEL_INFO["target_col"]
BEST_MODEL   = MODEL_INFO["best_model"]

# ── Load tất cả models vào memory khi server start ──────────────────────────
MODELS = {}

def load_all_models():
    """Load tất cả models đã train. Gọi 1 lần khi app start."""

    # Ridge (sklearn Pipeline, dùng joblib)
    ridge_path = "models/ridge.pkl"
    if os.path.exists(ridge_path):
        MODELS["ridge"] = joblib.load(ridge_path)
        print(f"Loaded Ridge  ← {ridge_path}")

    # XGBoost
    xgb_path = "models/xgboost.json"
    if os.path.exists(xgb_path):
        m = xgb.XGBRegressor()
        m.load_model(xgb_path)
        MODELS["xgboost"] = m
        print(f"Loaded XGBoost ← {xgb_path}")

    # LightGBM
    lgb_path = "models/lightgbm.txt"
    if os.path.exists(lgb_path):
        MODELS["lightgbm"] = lgb.Booster(model_file=lgb_path)
        print(f"Loaded LightGBM ← {lgb_path}")

    print(f"\nAvailable models: {list(MODELS.keys())}")
    print(f"Default (best)  : {BEST_MODEL}\n")


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Demand Forecasting API",
    description = "Predict daily sales demand for FreshRetailNet products.",
    version     = "1.0.0",
)


@app.on_event("startup")
def startup_event():
    load_all_models()


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    # Calendar
    day_of_week  : int   = Field(..., ge=0, le=6,   description="0=Monday, 6=Sunday")
    week_of_year : int   = Field(..., ge=1, le=53,  description="ISO week number")
    month        : int   = Field(..., ge=1, le=12)
    year         : int   = Field(..., ge=2020, le=2030)
    is_weekend   : int   = Field(..., ge=0, le=1)

    # Lag features (required — phải có data lịch sử)
    lag_1        : float = Field(..., ge=0, description="Sales yesterday")
    lag_7        : float = Field(..., ge=0, description="Sales same day last week")
    lag_14       : float = Field(..., ge=0, description="Sales 2 weeks ago")
    lag_28       : float = Field(..., ge=0, description="Sales 4 weeks ago")

    # Rolling features
    rolling_mean_7  : float = Field(..., ge=0)
    rolling_mean_14 : float = Field(..., ge=0)
    rolling_std_7   : float = Field(..., ge=0)

    # Price / Promo
    discount_pct : float = Field(..., ge=0.0, le=1.0, description="0=no discount, 0.5=50% off")
    is_promotion : int   = Field(..., ge=0, le=1)
    holiday      : int   = Field(..., ge=0, le=1)

    # Stockout
    stockout_lag_1 : float = Field(..., ge=0, le=1, description="Yesterday's stockout")

    # Category / Location
    product_category    : int = Field(..., description="first_category_id")
    product_subcategory : int = Field(..., description="second_category_id")
    location            : int = Field(..., description="city_id")
    store_id            : int = Field(...)

    # Weather
    avg_temperature : float = Field(..., description="Average temperature (°C)")
    avg_humidity    : float = Field(..., ge=0, le=100)
    avg_wind_level  : float = Field(..., ge=0)
    precpt          : float = Field(..., ge=0, description="Precipitation (mm)")

    # Optional: choose specific model
    model_name : Optional[str] = Field(
        default=None,
        description="Choose model: 'ridge', 'xgboost', 'lightgbm'. Default = best model."
    )

    @validator("model_name")
    def validate_model_name(cls, v):
        valid = {"ridge", "xgboost", "lightgbm", None}
        if v not in valid:
            raise ValueError(f"model_name must be one of: {valid}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "day_of_week"        : 2,
                "week_of_year"       : 15,
                "month"              : 4,
                "year"               : 2024,
                "is_weekend"         : 0,
                "lag_1"              : 0.3,
                "lag_7"              : 0.25,
                "lag_14"             : 0.2,
                "lag_28"             : 0.3,
                "rolling_mean_7"     : 0.27,
                "rolling_mean_14"    : 0.25,
                "rolling_std_7"      : 0.05,
                "discount_pct"       : 0.0,
                "is_promotion"       : 0,
                "holiday"            : 0,
                "stockout_lag_1"     : 0,
                "product_category"   : 5,
                "product_subcategory": 6,
                "location"           : 0,
                "store_id"           : 0,
                "avg_temperature"    : 22.5,
                "avg_humidity"       : 75.0,
                "avg_wind_level"     : 1.5,
                "precpt"             : 2.1,
                "model_name"         : None
            }
        }


class PredictResponse(BaseModel):
    prediction  : float = Field(..., description="Predicted units_ordered")
    model_used  : str
    input_echo  : dict  = Field(..., description="Input features đã nhận")


class BatchPredictRequest(BaseModel):
    """Predict nhiều rows cùng lúc."""
    requests : list[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions : list[float]
    model_used  : str
    count       : int


class HealthResponse(BaseModel):
    status        : str
    models_loaded : list[str]
    best_model    : str
    metrics       : dict


# ── Helper ───────────────────────────────────────────────────────────────────

def request_to_dataframe(req: PredictRequest) -> pd.DataFrame:
    data = {col: [getattr(req, col)] for col in FEATURE_COLS}
    return pd.DataFrame(data)


def get_model_and_name(model_name: Optional[str]):

    name_map = {
        "Ridge Regression": "ridge",
        "XGBoost"         : "xgboost",
        "LightGBM"        : "lightgbm",
    }
    key = model_name or name_map.get(BEST_MODEL, "lightgbm")

    if key not in MODELS:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{key}' chưa được load. Available: {list(MODELS.keys())}"
        )
    return MODELS[key], key


def run_inference(model, model_key: str, df: pd.DataFrame) -> np.ndarray:
    if model_key == "lightgbm" and isinstance(model, lgb.Booster):
        preds = model.predict(df)
    else:
        preds = model.predict(df)

    # Clip về 0 — không thể predict âm
    return np.clip(preds, 0, None)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Demand Forecasting API is running ",
        "docs"   : "/docs",
        "health" : "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status        = "ok",
        models_loaded = list(MODELS.keys()),
        best_model    = BEST_MODEL,
        metrics       = MODEL_INFO.get("metrics", {}),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    model, model_key = get_model_and_name(req.model_name)
    df = request_to_dataframe(req)

    try:
        preds = run_inference(model, model_key, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return PredictResponse(
        prediction = round(float(preds[0]), 4),
        model_used = model_key,
        input_echo = req.dict(exclude={"model_name"}),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(req: BatchPredictRequest):
    if len(req.requests) == 0:
        raise HTTPException(status_code=400, detail="requests list không được rỗng")
    if len(req.requests) > 500:
        raise HTTPException(status_code=400, detail="Tối đa 500 rows mỗi request")

    model, model_key = get_model_and_name(req.requests[0].model_name)

    dfs = [request_to_dataframe(r) for r in req.requests]
    df_all = pd.concat(dfs, ignore_index=True)

    try:
        preds = run_inference(model, model_key, df_all)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return BatchPredictResponse(
        predictions = [round(float(p), 4) for p in preds],
        model_used  = model_key,
        count       = len(preds),
    )


@app.get("/models", tags=["Models"])
def list_models():
    model_details = {}
    for key in MODELS:
        metrics = MODEL_INFO.get("metrics", {})
        model_details[key] = {
            "loaded"    : True,
            "MAE"       : metrics.get("MAE",   {}).get(key, "N/A"),
            "RMSE"      : metrics.get("RMSE",  {}).get(key, "N/A"),
            "MAPE%"     : metrics.get("MAPE%", {}).get(key, "N/A"),
            "is_best"   : (key == BEST_MODEL or
                           BEST_MODEL in {"Ridge Regression": "ridge",
                                          "XGBoost": "xgboost",
                                          "LightGBM": "lightgbm"}.get(BEST_MODEL, "")),
        }
    return {"models": model_details, "best_model": BEST_MODEL}


@app.get("/features", tags=["Models"])
def list_features():
    return {
        "feature_cols" : FEATURE_COLS,
        "target_col"   : TARGET_COL,
        "total_features": len(FEATURE_COLS),
    }


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)