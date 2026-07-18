import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set working directory to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# Load data - This path is correct!
train = pd.read_csv("data/train.csv")
stores = pd.read_csv("data/stores.csv")
transactions = pd.read_csv("data/transactions.csv")
holidays = pd.read_csv("data/holidays_events.csv")
oil = pd.read_csv("data/oil.csv")

# Display first few rows
print("=== First Few Rows ===")
print(train.head())
print(stores.head())
print(transactions.head())
print(holidays.head())
print(oil.head())

# Display shapes
print("\n=== Dataset Shapes ===")
print("Train:", train.shape)
print("Stores:", stores.shape)
print("Transactions:", transactions.shape)
print("Holidays:", holidays.shape)
print("Oil:", oil.shape)

# Display info
print("\n=== Dataset Info ===")
train.info()
stores.info()
transactions.info()
holidays.info()
oil.info()

# Display statistics
print("\n=== Train Statistics ===")
print(train.describe())

# Check missing values
print("\n=== Missing Values ===")
print("Train:", train.isnull().sum())
print("Stores:", stores.isnull().sum())
print("Transactions:", transactions.isnull().sum())
print("Holidays:", holidays.isnull().sum())
print("Oil:", oil.isnull().sum())

# Check duplicates
print("\n=== Duplicates ===")
print("Train:", train.duplicated().sum())
print("Stores:", stores.duplicated().sum())
print("Transactions:", transactions.duplicated().sum())
print("Holidays:", holidays.duplicated().sum())
print("Oil:", oil.duplicated().sum())

# Convert to datetime
train["date"] = pd.to_datetime(train["date"])
transactions["date"] = pd.to_datetime(transactions["date"])
holidays["date"] = pd.to_datetime(holidays["date"])
oil["date"] = pd.to_datetime(oil["date"])

print("\n=== Date Range ===")
print("Start Date:", train["date"].min())
print("End Date:", train["date"].max())

# Daily sales trend
daily_sales = train.groupby("date")["sales"].sum().reset_index()
print(daily_sales.head())

plt.figure(figsize=(15, 6))
plt.plot(daily_sales["date"], daily_sales["sales"])
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Top product families
top_family = train.groupby("family")["sales"].sum().sort_values(ascending=False).head(10)
top_family.plot(kind="bar", figsize=(12, 5))
plt.title("Top 10 Product Families by Sales")
plt.xlabel("Product Family")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Sales by store
store_sales = train.groupby("store_nbr")["sales"].sum()
store_sales.plot(figsize=(15, 5))
plt.title("Sales by Store")
plt.xlabel("Store Number")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Correlation matrix
numeric_data = train.select_dtypes(include=["number"])
corr = numeric_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()