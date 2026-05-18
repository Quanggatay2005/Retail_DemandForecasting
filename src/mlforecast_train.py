"""
src/mlforecast_train.py
=======================
Train demand forecasting models using mlforecast framework.

Run individually:
    python src/mlforecast_train.py --city_id 0 --horizon 7

"""

import argparse
import json
import os
import pickle
import time
import warnings
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd, ExpandingMean
from mlforecast.target_transforms import LocalStandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from window_ops.rolling import rolling_mean

warnings.filterwarnings("ignore")

MODELS_DIR    = "models/mlforecast"
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"



def prepare_mlforecast_data(
    city_id: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_path = f"{RAW_DIR}/train_clean.parquet"
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Not found {train_path}. "
        )

    df = pd.read_parquet(train_path)

    if city_id is not None:
        df = df[df["location"] == city_id].copy()
        print(f"Filtered city_id={city_id} → {len(df):,} rows")
    df["unique_id"] = (
        df["product_id"].astype(str) + "_" + df["store_id"].astype(str)
    )
    df = df.rename(columns={
        "date"          : "ds",
        "units_ordered" : "y",
    })

    # Calculate discount_pct if not exists
    if "discount_pct" not in df.columns:
        df["discount_pct"] = (1.0 - df["selling_price"]).clip(lower=0)

    # Exogenous variables
    EXOG_COLS = [
        "discount_pct", "is_promotion", "holiday",
        "avg_temperature", "avg_humidity", "avg_wind_level", "precpt",
        "stockout_flag",
    ]
    # Fill null cho exog cols
    for col in EXOG_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    # Chỉ giữ các cột cần thiết
    keep_cols = ["unique_id", "ds", "y"] + EXOG_COLS
    df = df[keep_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # ── Time-based split ─────────────────────────────────────────────────────
    # Dùng 80% đầu train, 20% cuối test — giống pipeline cũ
    cutoff_idx  = int(len(df) * 0.8)
    cutoff_date = df["ds"].iloc[cutoff_idx]

    df_train = df[df["ds"] <  cutoff_date].copy()
    df_test  = df[df["ds"] >= cutoff_date].copy()

    print(f"unique_id count : {df['unique_id'].nunique():,}")
    print(f"df_train        : {df_train.shape} "
          f"({df_train['ds'].min().date()} → {df_train['ds'].max().date()})")
    print(f"df_test         : {df_test.shape}  "
          f"({df_test['ds'].min().date()} → {df_test['ds'].max().date()})")

    return df_train, df_test, EXOG_COLS


# ── Step 2: Define MLForecast object ────────────────────────────────────

def build_mlforecast(horizon: int = 7) -> MLForecast:
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer

    models = [
        make_pipeline(SimpleImputer(), Ridge(alpha=1.0)),
        xgb.XGBRegressor(
            n_estimators     = 200,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        ),
        lgb.LGBMRegressor(
            n_estimators     = 200,
            num_leaves       = 63,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = -1,
        ),
    ]

    fcst = MLForecast(
        models     = models,
        freq       = "D",                    # daily frequency
        lags       = [1, 7, 14, 28],

        lag_transforms = {
            1 : [ExpandingMean()],           
            7 : [
                RollingMean(window_size=7),  
                RollingStd(window_size=7),   
            ],
            14: [RollingMean(window_size=14)],
        },

        date_features = ["dayofweek", "month", "year"],
        target_transforms = [LocalStandardScaler()],
        num_threads = 4,
    )

    print(f"MLForecast built — horizon={horizon}")
    print(f"Models: {[type(m).__name__ for m in models]}")
    return fcst


# ── Step 3: Train ────────────────────────────────────────────────────────────

def train_mlforecast(
    fcst     : MLForecast,
    df_train : pd.DataFrame,
    exog_cols: list[str],
) -> MLForecast:
    print("\nTraining MLForecast...")
    t0 = time.time()

    fcst.fit(
        df_train,
        id_col       = "unique_id",
        time_col     = "ds",
        target_col   = "y",
        static_features = [],
    )

    elapsed = time.time() - t0
    print(f"✓ Training done in {elapsed:.1f}s")
    return fcst


# ── Step 4: Predict ──────────────────────────────────────────────────────────

def predict_mlforecast(
    fcst       : MLForecast,
    horizon    : int,
    df_test_exog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    print(f"\nPredicting horizon={horizon} days...")

    preds = fcst.predict(
        h        = horizon,
        X_df     = df_test_exog,
    )
    for col in ["Pipeline", "XGBRegressor", "LGBMRegressor"]:
        if col in preds.columns:
            preds[col] = preds[col].clip(lower=0)

    print(f"Predictions shape: {preds.shape}")
    print(preds.head(3))
    return preds


# ── Step 5: Evaluate ─────────────────────────────────────────────────────────

def evaluate_mlforecast(
    preds    : pd.DataFrame,
    df_test  : pd.DataFrame,
    horizon  : int,
) -> pd.DataFrame:
    test_dates = sorted(df_test["ds"].unique())[:horizon]
    df_actual  = df_test[df_test["ds"].isin(test_dates)][["unique_id", "ds", "y"]]

    df_eval = preds.merge(df_actual, on=["unique_id", "ds"], how="inner")

    if df_eval.empty:
        print(f"  Pred dates  : {preds['ds'].min()} → {preds['ds'].max()}")
        print(f"  Test dates  : {df_test['ds'].min()} → {df_test['ds'].max()}")
        return pd.DataFrame()

    results = []
    model_cols = ["Pipeline", "XGBRegressor", "LGBMRegressor"]

    print(f"\n{'='*55}")
    print("  MLFORECAST MODEL COMPARISON")
    print(f"{'='*55}")

    for col in model_cols:
        if col not in df_eval.columns:
            continue

        y_true = df_eval["y"].values
        y_pred = df_eval[col].values

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mask = y_true > 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        print(f"  [{col:20s}] MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")
        results.append({
            "model"  : col,
            "MAE"    : round(mae,  4),
            "RMSE"   : round(rmse, 4),
            "MAPE%"  : round(mape, 2),
        })

    print(f"{'='*55}")
    return pd.DataFrame(results).set_index("model")


# ── Step 6: Cross Validation ────────────────────────────────────────

def run_cross_validation(
    fcst     : MLForecast,
    df_train : pd.DataFrame,
    horizon  : int = 7,
    n_windows: int = 3,
) -> pd.DataFrame:
    print(f"\nRunning {n_windows}-fold time series cross-validation...")
    t0 = time.time()

    cv_preds = fcst.cross_validation(
        df        = df_train,
        id_col    = "unique_id",
        time_col  = "ds",
        target_col= "y",
        h         = horizon,
        n_windows = n_windows,
        step_size = horizon,    # mỗi fold shift thêm H ngày
        static_features = [],
    )

    elapsed = time.time() - t0
    print(f"CV done in {elapsed:.1f}s")
    print(f"CV results shape: {cv_preds.shape}")

    # Tính CV metrics
    model_cols = ["Pipeline", "XGBRegressor", "LGBMRegressor"]
    print(f"\n{'='*55}")
    print("  CROSS-VALIDATION RESULTS")
    print(f"{'='*55}")

    cv_results = []
    for col in model_cols:
        if col not in cv_preds.columns:
            continue
        y_true = cv_preds["y"].values
        y_pred = cv_preds[col].clip(lower=0).values
        mae    = mean_absolute_error(y_true, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
        mask   = y_true > 0
        mape   = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        print(f"  [{col:20s}] MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")
        cv_results.append({"model": col, "CV_MAE": round(mae, 4),
                           "CV_RMSE": round(rmse, 4), "CV_MAPE%": round(mape, 2)})

    print(f"{'='*55}")
    return pd.DataFrame(cv_results).set_index("model")


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_mlforecast_training(
    city_id : Optional[int] = None,
    horizon : int = 7,
    run_cv  : bool = True,
) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Prepare data
    print("=" * 55)
    print("  STEP 1: Prepare data")
    print("=" * 55)
    df_train, df_test, exog_cols = prepare_mlforecast_data(city_id=city_id)

    # 2. Build MLForecast
    print("\n" + "=" * 55)
    print("  STEP 2: Build MLForecast")
    print("=" * 55)
    fcst = build_mlforecast(horizon=horizon)

    # 3. Cross Validation (optional)
    cv_results = None
    if run_cv:
        print("\n" + "=" * 55)
        print("  STEP 3: Cross Validation")
        print("=" * 55)
        cv_results = run_cross_validation(fcst, df_train, horizon=horizon, n_windows=3)

    # 4. Train
    print("\n" + "=" * 55)
    print("  STEP 4: Train on full data")
    print("=" * 55)
    fcst = train_mlforecast(fcst, df_train, exog_cols)

    # 5. Predict & Evaluate on test set
    print("\n" + "=" * 55)
    print("  STEP 5: Predict & Evaluate")
    print("=" * 55)
    
    # Create X_df required by mlforecast
    expected_future = fcst.make_future_dataframe(h=horizon)
    X_df = expected_future.merge(
        df_test[["unique_id", "ds"] + exog_cols], 
        on=["unique_id", "ds"], 
        how="left"
    )
    for col in exog_cols:
        X_df[col] = X_df[col].fillna(0)

    preds      = predict_mlforecast(fcst, horizon=horizon, df_test_exog=X_df)
    eval_df    = evaluate_mlforecast(preds, df_test, horizon=horizon)

    # 6. Save
    print("\n" + "=" * 55)
    print("  STEP 6: Save")
    print("=" * 55)

    # Save MLForecast object
    mlf_path = f"{MODELS_DIR}/mlf_model.pkl"
    with open(mlf_path, "wb") as f:
        pickle.dump(fcst, f)
    print(f"✓ Saved {mlf_path}")

    # Save predictions
    preds_path = f"{MODELS_DIR}/predictions.parquet"
    preds.to_parquet(preds_path, index=False)
    print(f"✓ Saved {preds_path}")

    # Save metrics
    model_info = {
        "framework"   : "mlforecast",
        "horizon"     : horizon,
        "city_id"     : city_id,
        "model_path"  : mlf_path,
        "metrics"     : eval_df.to_dict() if not eval_df.empty else {},
        "cv_metrics"  : cv_results.to_dict() if cv_results is not None else {},
    }

    if not eval_df.empty:
        best = eval_df["MAE"].idxmin()
        model_info["best_model"] = best
        print(f"\n✓ Best model by MAE: {best}")

    with open(f"{MODELS_DIR}/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"✓ Saved {MODELS_DIR}/model_info.json")

    if not eval_df.empty:
        eval_df.to_csv(f"{MODELS_DIR}/results.csv")

    return model_info


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with mlforecast")
    parser.add_argument(
        "--city_id", type=int, default=None,
        help="Filter by city_id (optional)"
    )
    parser.add_argument(
        "--horizon", type=int, default=7,
        help="Forecast horizon (default: 7)"
    )
    parser.add_argument(
        "--no_cv", action="store_true",
        help="Skip cross-validation (faster)"
    )
    args = parser.parse_args()

    run_mlforecast_training(
        city_id = args.city_id,
        horizon = args.horizon,
        run_cv  = not args.no_cv,
    )