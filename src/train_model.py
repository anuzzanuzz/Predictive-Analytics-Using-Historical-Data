# ==========================================
# Sales Forecasting - Model Training
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd

df = pd.read_csv("data/processed/feature_engineered_sales.csv")

print(df.columns.tolist())

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("data/processed/feature_engineered_sales.csv")

print("Original Shape :", df.shape)

# ---------------------------------------------------
# Use Sample (Fast Training)
# ---------------------------------------------------

df = df.sample(n=200000, random_state=42)

print("Sample Shape :", df.shape)

# ---------------------------------------------------
# Convert Date
# ---------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

# ---------------------------------------------------
# Label Encode Categorical Columns
# ---------------------------------------------------

categorical_columns = [
    "family",
    "city",
    "state",
    "store_type",
    "holiday_type",
    "locale",
    "description"
]

encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col].astype(str))

print("Categorical Encoding Completed!")

# ---------------------------------------------------
# Prepare Features
# ---------------------------------------------------

X = df.drop(columns=["id", "sales", "date"])

y = df["sales"]

print("Feature Matrix :", X.shape)
print("Target Shape   :", y.shape)

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

print("\nTraining Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed!")

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

predictions = model.predict(X_test)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")
print(f"R² Score  : {r2:.4f}")

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

joblib.dump(model, "models/sales_forecasting_model.pkl")

print("\nModel saved successfully!")
print("Location: models/sales_forecasting_model.pkl")

# ---------------------------------------------------
# Save Predictions
# ---------------------------------------------------

results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

results.to_csv(
    "outputs/predictions.csv",
    index=False
)

print("Predictions saved to outputs/predictions.csv")
print("\nProject Completed Successfully!")