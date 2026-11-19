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
