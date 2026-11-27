# Sales Forecasting AI

A simple machine learning tool that predicts future sales using historical data.  
This project demonstrates AI-based business forecasting using Linear Regression.

---

## 📊 Features
- Reads historical sales data (CSV)
- Trains a Linear Regression model
- Predicts future sales values
- Displays actual vs predicted values
- Optionally plots graph (future enhancement)

---

## 🧪 Example CSV Format

```
month,sales
1,5000
2,5500
3,6000
4,6200
5,6500
6,7000
```

---

## 🚀 Usage

### 1. Clone the repository
```bash
git clone https://github.com/Shashwat-Aneja/sales-forecasting-ai
cd sales-forecasting-ai
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn
```

### 3. Run the script
```bash
python forecast.py data.csv
```

---

## 📎 Project Structure

```
sales-forecasting-ai/
│
├── forecast.py
└── README.md
```

---

## 🔎 Example Output

```
📈 Sales Forecasting AI
-----------------------
Training model using historical data...
Predicted sales for next period: ₹7320.45
```

---

## 📌 Future Enhancements
- Add ARIMA / LSTM model
- Graph visualization using matplotlib
- Export forecast as JSON or CSV
- Integrate into XYLO business automation

---

Built by **Shashwat Aneja**

Forecast horizons are limited to 1,200 monthly periods (100 years) to bound allocation and output size. Larger horizons return a validation error.

All forecast entry points share the same history validation. Repeated DataFrame row indexes are normalized; boolean values and duplicate columns are rejected before fitting.

Historical averages and MAE metrics use scaled summation so finite observations do not produce infinite reports merely because their intermediate sum overflows.

Use `--json` for a machine-readable forecast array and diagnostic summary, including rolling backtest metrics when available.

Use `--output forecast.csv` to atomically export all requested forecast periods. It can be combined with `--json`; source-file aliases are rejected and a failed write preserves the previous export.

`--delimiter ";"`, `--delimiter tab` and `--delimiter "|"` support alternate input CSV separators, including quoted fields. Duplicate-header validation uses the selected separator. The default input and all CSV exports remain comma-separated.

Backtesting defaults to the latest 1,000 eligible held-out periods to bound repeated model fitting on large histories. Results disclose available/evaluated period counts and the first evaluated month. Programmatic callers can pass `backtest_forecast(history, max_periods=N)`. Each fold still trains only on earlier observations.

Seekable caller-owned text and UTF-8 binary CSV streams now receive the same duplicate-header checks as file paths, before pandas can rename ambiguous columns. Validation restores the starting position and leaves caller streams open; nonseekable stream objects are rejected explicitly.

Rolling backtests now evaluate a last-observation baseline on the identical held-out folds, reporting its MAE/RMSE and linear-model MAE skill percentage. Skill is unavailable when baseline MAE is zero or a comparison is not representable; negative skill means the linear model performed worse. These are historical diagnostics, not promises of future accuracy.

`--lookback-periods N` fits forecasts and diagnostics on the latest N observed periods, after normal history validation. JSON summaries disclose selected and available history counts. The default remains all history; observed periods need not be consecutive calendar months.

CSV paths ending in `.gz`, `.bz2`, or `.xz` are supported without unpacking a separate file. Header validation reads the same compression format as pandas and still rejects duplicate columns, including BOM-prefixed quoted headers and alternate delimiters. Archive containers such as ZIP/TAR are outside this feature.

`--backtest-periods N` bounds the number of latest held-out rolling evaluations (default 1000), independently of the training history window. Available/evaluated coverage and first evaluation month remain disclosed. N must be positive.

Forecast summaries disclose `history_period_span` and `missing_history_periods` within the selected observed window. Gaps are not imputed as zero sales; the last-observation comparator remains based on the last available observation, which may be more than one month earlier.

Corrupt compressed bodies, invalid DEFLATE blocks and header-parser limit failures use the normal validation-error path rather than escaping as EOFError, zlib.error or csv.Error. Header inspection restores caller stream position/ownership on these failures. No invalid input is exported.

### Optional raw CSV row budget

Use `--max-input-rows 100000` to reject an input file with more than that many data rows before modeling or export. The budget is per file, including comparison/training/scoring inputs, and applies before filtering or aggregation. The default remains unlimited. Values must be integers from 1 to 10,000,000. Only the budget plus one rows are loaded; oversized inputs fail rather than being silently truncated. This is a row limit, not a byte or field-size limit. Python callers can use `max_rows=`.

Forecast metric means retain small cancellation residuals and use an exact fallback when an intermediate sum overflows.

`--non-negative` clamps future and held-out trend predictions to zero and records the policy in JSON. Historical in-sample fit diagnostics remain the raw linear fit. Default forecasts preserve negative values for net-sales histories. This constraint is not a prediction guarantee.
