import os
import warnings
from typing import Tuple

import pandas as pd
import numpy as np
from datasets import load_dataset as hf_load_dataset

warnings.filterwarnings("ignore")

"""
Logic Process: 
1. Load dataset
2. Flatten
3. Clean
4. Save to raw/

1. Load:
    - Load dataset from HuggingFace
    - Filter by city_id to reduce RAM usage

2. Flatten:
    - Calculate hours_sale_sum, stockout_hours, stockout_flag
    - Rename columns
    - Drop unnecessary columns

3. Clean:
    - Drop duplicates
    - Fill null values
    - Remove negative values
    - Winsorize outliers

4. Save:
    - Save to raw/
"""
# ── Constants ────────────────────────────────────────────────────────────────
DATASET_NAME = "Dingdong-Inc/FreshRetailNet-50K"
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

# ── Step 1: Load ─────────────────────────────────────────────────────────────

def load_dataset(city_id: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading {DATASET_NAME}...")
    dataset = hf_load_dataset(DATASET_NAME)

    df_train = dataset["train"].to_pandas()
    df_eval  = dataset["eval"].to_pandas()

    if city_id is not None:
        df_train = df_train[df_train["city_id"] == city_id].copy()
        df_eval  = df_eval[df_eval["city_id"]   == city_id].copy()
        print(f"Filtered city_id={city_id}")

    print(f"Train : {df_train.shape}")
    print(f"Eval  : {df_eval.shape}")
    return df_train, df_eval


# ── Step 2: Flatten & Rename ─────────────────────────────────────────────────

def flatten_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df["hours_sale_sum"] = df["hours_sale"].apply(
        lambda x: float(sum(x)) if x is not None else 0.0
    )
    df["stockout_hours"] = df["hours_stock_status"].apply(
        lambda x: int(sum(x)) if x is not None else 0
    )
    df["stockout_flag"] = (df["stockout_hours"] > 0).astype(int)
    df = df.rename(columns={
        "dt"                 : "date",
        "sale_amount"        : "units_ordered",
        "stock_hour6_22_cnt" : "stock_on_hand",
        "discount"           : "selling_price",
        "activity_flag"      : "is_promotion",
        "holiday_flag"       : "holiday",
        "first_category_id"  : "product_category",
        "second_category_id" : "product_subcategory",
        "city_id"            : "location",
    })
    df = df.drop(columns=["hours_sale", "hours_stock_status"], errors="ignore")
    return df


# ── Step 3: Clean ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý:
    - Duplicate rows
    - Null values
    - Negative values
    - Outliers (winsorize, không xóa vì holiday/promo là signal thật)
    """
    original_len = len(df)

    # 3.1 Drop duplicates
    df = df.drop_duplicates(subset=["product_id", "store_id", "date"])
    print(f"  drop_duplicates : {original_len} → {len(df)} rows")

    # 3.2 Fill null
    df["units_ordered"] = df["units_ordered"].fillna(0)
    df["selling_price"] = df["selling_price"].fillna(1.0)
    df["is_promotion"]  = df["is_promotion"].fillna(0).astype(int)
    df["holiday"]       = df["holiday"].fillna(0).astype(int)

    # Sắp xếp theo location và date để đảm bảo tính liên tục của time-series
    df = df.sort_values(["location", "date"])
    
    for col in ["avg_temperature", "avg_humidity", "avg_wind_level", "precpt"]:
        if col in df.columns:
            # 1. Nội suy tuyến tính (Linear Interpolation) theo từng thành phố
            # 2. Forward fill và Backward fill cho những ngày ở rìa (đầu/cuối) bị thiếu
            df[col] = df.groupby("location")[col].transform(
                lambda x: x.interpolate(method='linear').ffill().bfill()
            )

    # 3.3 Remove negative values
    df = df[df["units_ordered"] >= 0]
    df = df[df["selling_price"] > 0]

    # 3.4 Winsorize outlier 
    q1  = df["units_ordered"].quantile(0.01)
    q99 = df["units_ordered"].quantile(0.99)
    cap = q99 + 3 * (q99 - q1)
    n_outliers = (df["units_ordered"] > cap).sum()
    df["units_ordered"] = df["units_ordered"].clip(upper=cap)
    print(f"  winsorize       : {n_outliers} rows capped tại {cap:.3f}")

    print(f"  final shape     : {df.shape}")
    return df


# ── main pipeline ────────────────────────────────────────────────────────────

def run_data_preparation(city_id: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(RAW_DIR, exist_ok=True)

    # Load
    df_train, df_eval = load_dataset(city_id=city_id)

    # Flatten + rename
    print("\nFlatten & rename...")
    df_train = flatten_and_rename(df_train)
    df_eval  = flatten_and_rename(df_eval)

    # Clean
    print("\nCleaning train...")
    df_train = clean_data(df_train)

    print("\nCleaning eval...")
    df_eval = clean_data(df_eval)

    # Save
    df_train.to_parquet(f"{RAW_DIR}/train_clean.parquet", index=False)
    df_eval.to_parquet(f"{RAW_DIR}/eval_clean.parquet",   index=False)
    print(f"\n✓ Saved to {RAW_DIR}/")

    return df_train, df_eval


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data preparation pipeline")
    parser.add_argument("--city_id", type=int, default=None,
                        help="Filter by city_id")
    args = parser.parse_args()

    run_data_preparation(city_id=args.city_id)