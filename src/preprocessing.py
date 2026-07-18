import pandas as pd
import numpy as np
from pathlib import Path

# Make file paths independent of the current working directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
stores = pd.read_csv(DATA_DIR / "stores.csv")
transactions = pd.read_csv(DATA_DIR / "transactions.csv")
holidays = pd.read_csv(DATA_DIR / "holidays_events.csv")
oil = pd.read_csv(DATA_DIR / "oil.csv")

#convert date columns to datetime format
train["date"] = pd.to_datetime(train["date"])
transactions["date"] = pd.to_datetime(transactions["date"])
holidays["date"] = pd.to_datetime(holidays["date"])
oil["date"] = pd.to_datetime(oil["date"])

train = train.drop_duplicates()
stores = stores.drop_duplicates()
transactions = transactions.drop_duplicates()
holidays = holidays.drop_duplicates()
oil = oil.drop_duplicates()

# Rename store type column so holiday type can merge without conflict
stores = stores.rename(columns={"type": "store_type"})

print(train.duplicated().sum())

print(train.isnull().sum())
print(stores.isnull().sum())
print(transactions.isnull().sum())
print(holidays.isnull().sum())
print(oil.isnull().sum())
oil["dcoilwtico"] = oil["dcoilwtico"].ffill()#Fill missing oil prices using forward fill: 
oil["dcoilwtico"] = oil["dcoilwtico"].bfill()#missing values remain at the beginning, fill them with backward fill:
oil.isnull().sum()

holidays = holidays[holidays["transferred"] == False]#to handle the holidays only keep active holidays
print(holidays.shape)
#merge the store information#
df = train.merge(
    stores,
    on="store_nbr",
    how="left"
)
df.head()
#merge transaction
df = df.merge(
    transactions,
    on=["date", "store_nbr"],
    how="left"
)
#merge oil prices
df = df.merge(
    oil,
    on="date",
    how="left"
)
# Merge Holiday Information
holiday_data = holidays[
    ["date", "type", "locale", "description"]
].rename(columns={"type": "holiday_type"})

df = df.merge(
    holiday_data,
    on="date",
    how="left"
)

# Rows without holidays will have missing values; fill holiday columns with no-holiday defaults.
df["holiday_type"] = df["holiday_type"].fillna("No Holiday")
df["locale"] = df["locale"].fillna("None")
df["description"] = df["description"].fillna("No Holiday")
df["transactions"] = df["transactions"].fillna(0)#Missing transactions (e.g., closed stores) can be set to 0:
df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()#Fill any remaining missing oil prices:

df.isnull().sum()
print(df.shape)

df.to_csv(PROCESSED_DIR / "cleaned_sales.csv", index=False)

cleaned = pd.read_csv(PROCESSED_DIR / "cleaned_sales.csv")
cleaned.head()