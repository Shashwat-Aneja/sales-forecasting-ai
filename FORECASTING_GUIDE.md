Sales forecasting guide

Input CSV must contain numeric `month` and `sales` columns. Each month index must be unique, finite, and at least two distinct periods must be present.

Run `python forecast.py sample_sales_history.csv` after installing `requirements.txt`.

The model is a linear-trend baseline that predicts one period beyond the latest month. It does not model seasonality, promotions, stockouts, holidays, or structural changes. Before operational use, compare against a naive baseline on a chronological holdout and report forecast error.