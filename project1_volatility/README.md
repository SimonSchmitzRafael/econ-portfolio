# AAPL Volatility Forecasting with GARCH

Analyzing and forecasting Apple stock volatility using GARCH(1,1), with an
honest evaluation of forecast accuracy across different horizons.

## Key findings

- **Prices are non-stationary, returns are stationary** (ADF test:
  price p-value = 0.78, return p-value ≈ 0), confirming returns —
  not raw prices — are the correct basis for time-series modeling.
- **Strong volatility clustering** is present and statistically confirmed:
  a GARCH(1,1) model estimates α = 0.096, β = 0.872 (α+β = 0.968),
  meaning ~87% of volatility persists day to day, with COVID-19 (2020)
  and a 2025 event as the two clearest volatility spikes in the data.
- **Forecast accuracy is highly horizon-dependent.** A 1-day-ahead
  rolling GARCH forecast (RMSE 0.00295) clearly outperforms both a naive
  constant-mean baseline (0.00617) and a 151-day multi-step forecast
  (0.00801) — showing GARCH's value lies in short-horizon reactivity,
  not long-horizon projection.
- **A lag-1 return regression** initially suggests mild, significant
  mean-reversion (coef = -0.087, p = 0.001), but this significance
  disappears under Newey-West (HAC) standard errors (p = 0.120) —
  the same volatility clustering documented above was distorting the
  naive standard errors.

## Method

1. Pulled 6 years of AAPL daily price data (`yfinance`), stored in SQLite.
2. Computed daily returns; confirmed stationarity via Augmented Dickey-Fuller test.
3. Measured volatility clustering via 20-day rolling standard deviation.
4. Fit GARCH(1,1) via maximum likelihood; compared to rolling-std estimate.
5. Evaluated forecasts out-of-sample (90/10 train/test split): multi-step
   vs. 1-day-ahead rolling, benchmarked against a naive baseline (RMSE).
6. Ran a lag-1 OLS regression on returns; tested for serial correlation
   (Breusch-Godfrey) and applied Newey-West standard errors.

## Files

- `01_data_collection.ipynb` — full analysis notebook (data pull through
  final regression)
- `volatility_project.db` — SQLite database with price/return/volatility data

## Tools

Python, pandas, numpy, matplotlib, statsmodels, arch, scikit-learn, SQLite