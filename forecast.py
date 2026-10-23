import math
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression

REQUIRED_COLUMNS = {"month", "sales"}


def load_sales_history(csv_file):
    """Load and validate monthly sales data, returning it in month order."""
    df = pd.read_csv(csv_file)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(REQUIRED_COLUMNS))}")

    df = df.loc[:, ["month", "sales"]].copy()
    for column in ("month", "sales"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df.empty or df.isna().any().any():
        raise ValueError("Sales history must contain numeric month and sales values.")
    if not all(math.isfinite(float(value)) for value in df.to_numpy().ravel()):
        raise ValueError("Month and sales values must be finite numbers.")
    if df["month"].nunique() < 2:
        raise ValueError("At least two distinct months are required for a forecast.")
    if df["month"].duplicated().any():
        raise ValueError("Each month must appear only once in the sales history.")
    return df.sort_values("month").reset_index(drop=True)


def predict_next_sales(df):
    """Fit a linear trend and predict sales for the period after the latest month."""
    model = LinearRegression()
    model.fit(df[["month"]], df["sales"])
    next_month = float(df["month"].iloc[-1]) + 1
    return float(model.predict(pd.DataFrame({"month": [next_month]}))[0])


def summarize_forecast(history, predicted_sales):
    """Return transparent context for a one-period sales forecast."""
    latest_month = float(history["month"].max())
    latest_sales = float(history.loc[history["month"].idxmax(), "sales"])
    if predicted_sales > latest_sales:
        trend = "increasing"
    elif predicted_sales < latest_sales:
        trend = "decreasing"
    else:
        trend = "stable"
    return {
        "next_month": latest_month + 1,
        "historical_average": float(history["sales"].mean()),
        "predicted_sales": float(predicted_sales),
        "trend": trend,
    }


def forecast_sales(csv_file):
    try:
        history = load_sales_history(csv_file)
        predicted_sales = predict_next_sales(history)
        print("\n📈 Sales Forecasting AI")
        print("-----------------------")
        print("Training model using historical data...")
        summary = summarize_forecast(history, predicted_sales)
        print(f"Predicted sales for next period: ₹{predicted_sales:.2f}")
        print(f"Historical average: ₹{summary['historical_average']:.2f}")
        print(f"Trend: {summary['trend']}\n")
        return predicted_sales
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"❌ Error: {exc}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python forecast.py <csv_file>")
        return 2
    return 0 if forecast_sales(sys.argv[1]) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
