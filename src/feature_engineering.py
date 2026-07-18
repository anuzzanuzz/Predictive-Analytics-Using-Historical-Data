# Feature Engineering for Sales Forecasting Project

import pandas as pd
import numpy as np


from pathlib import Path

# -----------------------------
# Step 1: Load Cleaned Dataset
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

df = pd.read_csv(PROCESSED_DIR / "cleaned_sales.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])


# -----------------------------
# Step 2: Sort Data
# -----------------------------

df = df.sort_values(
    ["store_nbr", "family", "date"]
)


# -----------------------------
# Step 3: Date Features
# -----------------------------

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day"] = df["date"].dt.day

df["weekday"] = df["date"].dt.weekday

df["quarter"] = df["date"].dt.quarter


# Weekend feature
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)


# -----------------------------
# Step 4: Time Index Feature
# -----------------------------

df["time_index"] = (
    (df["year"] - df["year"].min()) * 12
    + df["month"]
)


# -----------------------------
# Step 5: Lag Features
# -----------------------------

# Previous day sales
df["lag_1"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .shift(1)
)


# Previous 7 days sales
df["lag_7"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .shift(7)
)


# Previous 30 days sales
df["lag_30"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .shift(30)
)


# -----------------------------
# Step 6: Rolling Average Features
# -----------------------------

df["rolling_7"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .shift(1)
    .rolling(window=7)
    .mean()
)


df["rolling_30"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .shift(1)
    .rolling(window=30)
    .mean()
)


# -----------------------------
# Step 7: Holiday Feature
# -----------------------------

# Handle old cleaned datasets that may have duplicate-suffix columns from merge
if "holiday_type" not in df.columns and "type_y" in df.columns:
    df = df.rename(columns={"type_y": "holiday_type"})
if "holiday_type" not in df.columns and "type" in df.columns:
    df = df.rename(columns={"type": "holiday_type"})

if "holiday_type" not in df.columns:
    raise KeyError("Expected holiday_type column in the dataset.")

df["is_holiday"] = (
    df["holiday_type"] != "No Holiday"
).astype(int)


# -----------------------------
# Step 8: Remove Missing Values
# -----------------------------

df = df.dropna()


# -----------------------------
# Step 9: Save Final Dataset
# -----------------------------

OUTPUT_PATH = PROCESSED_DIR / "feature_engineered_sales.csv"
df.to_csv(OUTPUT_PATH, index=False)


# -----------------------------
# Step 10: Check Output
# -----------------------------

print("Feature Engineering Completed!")

print("Dataset Shape:")
print(df.shape)

print("\nNew Features:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())