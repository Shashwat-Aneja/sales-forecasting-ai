import math

import numpy as np
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


def predict_sales_horizon(history, periods=1):
    """Predict one or more future monthly sales values from the fitted trend."""
    if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
        raise ValueError("periods must be a positive integer.")
    missing = REQUIRED_COLUMNS.difference(history.columns)
    if missing:
        raise ValueError(f"History must contain columns: {', '.join(sorted(REQUIRED_COLUMNS))}")
    ordered = history.loc[:, ["month", "sales"]].copy()
    for column in ("month", "sales"):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    if ordered.empty or ordered.isna().any().any():
        raise ValueError("History must contain numeric month and sales values.")
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("Month and sales values must be finite numbers.")
    if ordered["month"].duplicated().any():
        raise ValueError("Each month must appear only once in the sales history.")
    if ordered["month"].nunique() < 2:
        raise ValueError("At least two distinct months are required for a forecast.")
    ordered = ordered.sort_values("month").reset_index(drop=True)
    model = LinearRegression()
    model.fit(ordered[["month"]], ordered["sales"])
    first_future_month = float(ordered["month"].iloc[-1]) + 1
    future_months = np.arange(first_future_month, first_future_month + periods, dtype=float)
    predictions = model.predict(pd.DataFrame({"month": future_months}))
    return pd.DataFrame({"month": future_months, "predicted_sales": predictions.astype(float)})


def backtest_forecast(history):
    """Evaluate one-period-ahead predictions using expanding historical windows.

    At least three observations are required so that the first evaluation has
    two training periods and one held-out period. The function never trains on
    the observation it evaluates. Validate inputs here as callers may invoke
    backtesting directly without going through load_sales_history().
    """
    missing = REQUIRED_COLUMNS.difference(history.columns)
    if missing:
        raise ValueError(
            f"History must contain columns: {', '.join(sorted(REQUIRED_COLUMNS))}"
        )

    ordered = history.loc[:, ["month", "sales"]].copy()
    for column in ("month", "sales"):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    if ordered.empty or ordered.isna().any().any():
        raise ValueError("History must contain numeric month and sales values.")
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("Month and sales values must be finite numbers.")
    if ordered["month"].duplicated().any():
        raise ValueError("Each month must appear only once in the sales history.")

    ordered = ordered.sort_values("month").reset_index(drop=True)
    if len(ordered) < 3:
        return None

    errors = []
    for split_index in range(2, len(ordered)):
        training = ordered.iloc[:split_index]
        held_out = ordered.iloc[split_index]
        model = LinearRegression()
        model.fit(training[["month"]], training["sales"])
        prediction = float(
            model.predict(pd.DataFrame({"month": [float(held_out["month"])]}))[0]
        )
        errors.append(float(held_out["sales"]) - prediction)

    absolute_errors = np.abs(np.asarray(errors, dtype=float))
    return {
        "backtest_periods": len(errors),
        "backtest_mae": float(np.mean(absolute_errors)),
        "backtest_rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def summarize_forecast(history, predicted_sales):
    """Return transparent context and out-of-sample backtest metrics when possible."""
    latest_month = float(history["month"].max())
    latest_sales = float(history.loc[history["month"].idxmax(), "sales"])

    # These are in-sample fit diagnostics, not a guarantee of future forecast accuracy.
    fit_model = LinearRegression()
    fit_model.fit(history[["month"]], history["sales"])
    residuals = history["sales"].to_numpy(dtype=float) - fit_model.predict(history[["month"]])
    fit_mae = float(np.mean(np.abs(residuals)))
    fit_rmse = float(np.sqrt(np.mean(np.square(residuals))))

    if predicted_sales > latest_sales:
        trend = "increasing"
    elif predicted_sales < latest_sales:
        trend = "decreasing"
    else:
        trend = "stable"
    summary = {
        "next_month": latest_month + 1,
        "historical_average": float(history["sales"].mean()),
        "predicted_sales": float(predicted_sales),
        "trend": trend,
        "historical_fit_mae": fit_mae,
        "historical_fit_rmse": fit_rmse,
    }
    backtest = backtest_forecast(history)
    if backtest is not None:
        summary.update(backtest)
    return summary


def forecast_sales(csv_file, horizon=1):
    try:
        history = load_sales_history(csv_file)
        forecast = predict_sales_horizon(history, periods=horizon)
        predicted_sales = float(forecast["predicted_sales"].iloc[0])
        print("\n📈 Sales Forecasting AI")
        print("-----------------------")
        print("Training model using historical data...")
        summary = summarize_forecast(history, predicted_sales)
        print(f"Predicted sales for next period: ₹{predicted_sales:.2f}")
        print(f"Historical average: ₹{summary['historical_average']:.2f}")
        print(f"Trend: {summary['trend']}")
        print(f"Historical fit MAE (in-sample): ₹{summary['historical_fit_mae']:.2f}")
        print(f"Historical fit RMSE (in-sample): ₹{summary['historical_fit_rmse']:.2f}")
        if "backtest_mae" in summary:
            print(
                f"Rolling backtest MAE ({summary['backtest_periods']} periods): "
                f"₹{summary['backtest_mae']:.2f}"
            )
            print(f"Rolling backtest RMSE: ₹{summary['backtest_rmse']:.2f}\n")
        else:
            print("Rolling backtest: unavailable (requires at least 3 observed periods)\n")
        if horizon > 1:
            print("Extended forecast:")
            for row in forecast.itertuples(index=False):
                print(f"Month {row.month:g}: ₹{row.predicted_sales:.2f}")
        return predicted_sales
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"❌ Error: {exc}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Forecast monthly sales from a CSV history.")
    parser.add_argument("csv_file", help="CSV containing month and sales columns")
    parser.add_argument("--horizon", type=int, default=1, help="Number of future months to forecast")
    args = parser.parse_args()
    return 0 if forecast_sales(args.csv_file, horizon=args.horizon) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
