import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/processed/feature_engineered_sales.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# Encode Categorical Columns
# -----------------------------

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

# -----------------------------
# Features & Target
# -----------------------------

X = df.drop(columns=["id", "sales", "date"])
y = df["sales"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("models/sales_forecasting_model.pkl")

# -----------------------------
# Prediction
# -----------------------------

predictions = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# -----------------------------
# Save Predictions
# -----------------------------

results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

results.to_csv(
    "outputs/predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")

# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values[:200],
    label="Actual"
)

plt.plot(
    predictions[:200],
    label="Predicted"
)

plt.title("Actual vs Predicted Sales")

plt.xlabel("Samples")

plt.ylabel("Sales")

plt.legend()

plt.show()

# -----------------------------
# Feature Importance
# -----------------------------

if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 10 Important Features")
    print(importance.head(10))

    plt.figure(figsize=(10,6))

    plt.barh(
        importance["Feature"][:10],
        importance["Importance"][:10]
    )

    plt.title("Top 10 Feature Importance")

    plt.gca().invert_yaxis()

    plt.show()