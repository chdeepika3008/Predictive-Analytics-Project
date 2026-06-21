# forecasting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("../data/sales.csv")

print("\nDataset:")
print(df)

# ----------------------------
# Data Preprocessing
# ----------------------------
df["Month_Number"] = range(1, len(df) + 1)

print("\nProcessed Dataset:")
print(df)

# ----------------------------
# Features and Target
# ----------------------------
X = df[["Month_Number"]]
y = df["Sales"]

# ----------------------------
# Train Model
# ----------------------------
model = LinearRegression()
model.fit(X, y)

# ----------------------------
# Predictions on Existing Data
# ----------------------------
y_pred = model.predict(X)

# ----------------------------
# Future Forecast (Next 6 Months)
# ----------------------------
future_months = pd.DataFrame({
    "Month_Number": [13, 14, 15, 16, 17, 18]
})

future_predictions = model.predict(future_months)

print("\nFuture Sales Forecast:")
for month, sales in zip(future_months["Month_Number"], future_predictions):
    print(f"Month {month}: {sales:.2f}")

# ----------------------------
# Model Evaluation
# ----------------------------
score = r2_score(y, y_pred)

print(f"\nR² Score: {score:.4f}")

# ----------------------------
# Visualization
# ----------------------------
plt.figure(figsize=(10, 6))

# Actual Sales
plt.scatter(
    df["Month_Number"],
    df["Sales"],
    label="Actual Sales"
)

# Regression Line
plt.plot(
    df["Month_Number"],
    y_pred,
    linewidth=2,
    label="Regression Line"
)

# Future Predictions
plt.scatter(
    future_months["Month_Number"],
    future_predictions,
    marker="x",
    s=100,
    label="Future Predictions"
)

plt.title("Sales Forecasting Using Linear Regression")
plt.xlabel("Month Number")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

# Save Image
plt.savefig("../images/sales_forecast.png")

plt.show()

print("\nForecast graph saved successfully!")
print("Location: images/sales_forecast.png")