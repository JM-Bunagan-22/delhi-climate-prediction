"""
Delhi Daily Climate — Predictive Analysis
Predicts mean temperature (meantemp) using date-based and weather features.
Trained on DailyDelhiClimateTrain.csv, validated on DailyDelhiClimateTest.csv.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
train = pd.read_csv("DailyDelhiClimateTrain.csv", parse_dates=["date"])
test = pd.read_csv("DailyDelhiClimateTest.csv", parse_dates=["date"])

train = train.sort_values("date").reset_index(drop=True)
test = test.sort_values("date").reset_index(drop=True)

print(f"Train range: {train['date'].min().date()} to {train['date'].max().date()} ({len(train)} rows)")
print(f"Test range:  {test['date'].min().date()} to {test['date'].max().date()} ({len(test)} rows)")

# ---------------------------------------------------------------
# 2. Clean known data-quality issue: meanpressure has extreme outliers
#    (raw values include -3 and 7679 hPa; realistic sea-level pressure
#    is roughly 950-1050 hPa). Clip instead of dropping rows so the
#    time series stays continuous.
# ---------------------------------------------------------------
for df in (train, test):
    df["meanpressure"] = df["meanpressure"].clip(950, 1050)

# ---------------------------------------------------------------
# 3. Feature engineering
#    - Cyclical encoding of day-of-year and month (captures seasonality
#      without a hard break between Dec 31 and Jan 1)
#    - Calendar features
#    - Other same-day weather readings (humidity, wind, pressure)
# ---------------------------------------------------------------
def add_features(df):
    df = df.copy()
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["year"] = df["date"].dt.year
    return df

train_fe = add_features(train)
test_fe = add_features(test)

FEATURES = ["doy_sin", "doy_cos", "month_sin", "month_cos", "year",
            "humidity", "wind_speed", "meanpressure"]
TARGET = "meantemp"

X_train, y_train = train_fe[FEATURES], train_fe[TARGET]
X_test, y_test = test_fe[FEATURES], test_fe[TARGET]

# ---------------------------------------------------------------
# 4. Train candidate models
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                     learning_rate=0.05, random_state=42),
}

results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions[name] = preds
    results[name] = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }

results_df = pd.DataFrame(results).T.sort_values("RMSE")
print("\nModel comparison on test set:")
print(results_df.round(3))

best_name = results_df.index[0]
best_preds = predictions[best_name]
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(train["date"], train["meantemp"], label="Train (actual)", color="steelblue", alpha=0.7)
plt.plot(test["date"], test["meantemp"], label="Test (actual)", color="black")
plt.plot(test["date"], best_preds, label=f"Test (predicted - {best_name})",
         color="crimson", linestyle="--")
plt.title("Delhi Mean Temperature: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_vs_actual.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color="teal")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, color="gray", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual meantemp (°C)")
plt.ylabel("Predicted meantemp (°C)")
plt.title(f"Predicted vs Actual — {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual_scatter.png", dpi=150)
plt.close()

if best_name in ("Random Forest", "Gradient Boosting"):
    importances = pd.Series(models[best_name].feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(7, 4))
    importances.plot(kind="barh", color="darkorange")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

results_df.round(3).to_csv("model_comparison.csv")
print("\nSaved: forecast_vs_actual.png, predicted_vs_actual_scatter.png, feature_importance.png, model_comparison.csv")
