# Delhi Daily Climate — Temperature Prediction

Predicts daily mean temperature in Delhi from calendar-based seasonality and same-day weather readings, trained on 4 years of daily data and validated on a held-out future period.

## Problem

Given historical daily weather records (temperature, humidity, wind speed, pressure), can we predict `meantemp` for a period the model has never seen?

- **Train:** 2013-01-01 → 2017-01-01 (1,462 days)
- **Test:** 2017-01-01 → 2017-04-24 (114 days) — chronologically after train, used only for validation

Data source: [Daily Delhi Climate Data, Kaggle](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)

## Approach

1. **Data cleaning** — `meanpressure` contains invalid readings (raw range: -3 to 7,679 hPa); clipped to a realistic 950–1050 hPa band rather than dropping rows, to keep the daily series continuous.
2. **Feature engineering** — cyclical (sin/cos) encoding of day-of-year and month so Dec 31 and Jan 1 are treated as adjacent, plus year (trend) and same-day humidity/wind/pressure.
3. **Modeling** — compared Linear Regression, Random Forest, and Gradient Boosting regressors.
4. **Validation** — trained only on train data, scored strictly on the unseen test period.

## Results

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| **Linear Regression** | **2.11** | **2.56** | **0.84** |
| Gradient Boosting | 2.13 | 2.74 | 0.81 |
| Random Forest | 2.19 | 2.80 | 0.81 |

The seasonal cycle is strongly periodic, so a linear model on cyclically-encoded date features performs as well as more complex tree ensembles — extra model complexity doesn't add much here.

![Actual vs Predicted](forecast_vs_actual.png)

## Repo contents

```
├── climate_prediction.ipynb     # Full analysis notebook (run top to bottom)
├── analysis.py                  # Same pipeline as a standalone script
├── DailyDelhiClimateTrain.csv
├── DailyDelhiClimateTest.csv
├── forecast_vs_actual.png
├── predicted_vs_actual_scatter.png
├── model_comparison.csv
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python analysis.py
# or open climate_prediction.ipynb
```

## Possible next steps

- Add lagged temperature (previous 1–7 days) and rolling averages to capture short-term momentum
- Try a dedicated time-series model (SARIMA, Prophet) since forecasting typically won't have same-day humidity/wind/pressure available in advance — this version assumes those are known, which is realistic for nowcasting but not for a true multi-day-ahead forecast
