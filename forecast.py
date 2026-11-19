import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

REQUIRED_COLUMNS = {"month", "sales"}
MAX_SAFE_MONTH_INDEX = 2**53 - 1
MAX_FORECAST_PERIODS = 1200  # Bound memory use to at most 100 years of monthly output.


def _finite_predictions(values):
    """Do not publish invalid extrapolations even when all inputs are finite."""
    predictions = np.asarray(values, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Forecast results exceed the supported numeric range.")
    return predictions


def _stable_mean(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Forecast metrics require non-empty finite values.")
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    return scale * (math.fsum(float(value) for value in values / scale) / len(values))


def _stable_rmse(errors):
    """Compute finite RMSE without overflowing when finite residuals are squared."""
    values = np.asarray(errors, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Forecast residuals exceed the supported numeric range.")
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale == 0:
        return 0.0
    return scale * float(np.sqrt(np.mean(np.square(values / scale))))


def _validate_month_index(values):
    """Require whole-number period indexes because forecasts advance by one month."""
    months = np.asarray(values, dtype=float)
    if np.any(np.abs(months) > MAX_SAFE_MONTH_INDEX):
        raise ValueError("Month indexes exceed exact floating-point integer precision.")
    if not np.equal(months, np.floor(months)).all():
        raise ValueError("Month values must be whole-number period indexes.")


def validate_history(history, minimum_rows=2):
    """Normalize every public entry point, including duplicated DataFrame indexes."""
    if not isinstance(history, pd.DataFrame) or history.columns.duplicated().any():
        raise ValueError("History must be a DataFrame with unique column names.")
    if not REQUIRED_COLUMNS.issubset(history.columns):
        raise ValueError("History must contain columns: month, sales")
    ordered = history.loc[:, ["month", "sales"]].copy()
    if ordered.apply(lambda column: column.map(lambda value: isinstance(value, (bool, np.bool_)))).any().any():
        raise ValueError("History must contain numeric values, not booleans.")
    if any(pd.api.types.is_complex_dtype(ordered[column]) for column in ordered):
        raise ValueError("History must contain real numbers, not complex values.")
    for column in ("month", "sales"):
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    if ordered.empty or ordered.isna().any().any():
        raise ValueError("History must contain numeric month and sales values.")
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("Month and sales values must be finite numbers.")
    _validate_month_index(ordered["month"].to_numpy())
    if ordered["month"].duplicated().any():
        raise ValueError("Each month must appear only once in the sales history.")
    if len(ordered) < minimum_rows:
        raise ValueError("At least two distinct months are required for a forecast.")
    return ordered.sort_values("month").reset_index(drop=True)


def load_sales_history(csv_file):
    """Load and validate monthly sales data, returning it in month order."""
    return validate_history(pd.read_csv(csv_file))


def predict_next_sales(df):
    """Fit a linear trend and predict the period after the latest observed month."""
    ordered = validate_history(df)
    model = LinearRegression()
    origin = float(ordered["month"].iloc[0])
    model.fit(ordered[["month"]] - origin, ordered["sales"])
    if float(ordered["month"].iloc[-1]) >= MAX_SAFE_MONTH_INDEX:
        raise ValueError("Next month exceeds exact floating-point integer precision.")
    next_month = float(ordered["month"].iloc[-1]) + 1
    return float(_finite_predictions(model.predict(pd.DataFrame({"month": [next_month - origin]})))[0])


def predict_sales_horizon(history, periods=1):
    """Predict one or more future monthly sales values from the fitted trend."""
    if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
        raise ValueError("periods must be a positive integer.")
    if periods > MAX_FORECAST_PERIODS:
        raise ValueError(f"Forecast horizon cannot exceed {MAX_FORECAST_PERIODS} months.")
    ordered = validate_history(history)
    model = LinearRegression()
    origin = float(ordered["month"].iloc[0])
    model.fit(ordered[["month"]] - origin, ordered["sales"])
    if float(ordered["month"].iloc[-1]) + periods > MAX_SAFE_MONTH_INDEX:
        raise ValueError("Forecast horizon exceeds exact floating-point integer precision.")
    first_future_month = float(ordered["month"].iloc[-1]) + 1
    future_months = np.arange(first_future_month, first_future_month + periods, dtype=float)
    predictions = _finite_predictions(model.predict(pd.DataFrame({"month": future_months - origin})))
    return pd.DataFrame({"month": future_months, "predicted_sales": predictions.astype(float)})


def backtest_forecast(history):
    """Evaluate one-period-ahead predictions using expanding historical windows.

    At least three observations are required so that the first evaluation has
    two training periods and one held-out period. The function never trains on
    the observation it evaluates. Validate inputs here as callers may invoke
    backtesting directly without going through load_sales_history().
    """
    ordered = validate_history(history, minimum_rows=1)
    if len(ordered) < 3:
        return None

    errors = []
    for split_index in range(2, len(ordered)):
        training = ordered.iloc[:split_index]
        held_out = ordered.iloc[split_index]
        model = LinearRegression()
        origin = float(training["month"].iloc[0])
        model.fit(training[["month"]] - origin, training["sales"])
        prediction = float(
            _finite_predictions(model.predict(pd.DataFrame({"month": [float(held_out["month"]) - origin]})))[0]
        )
        errors.append(float(held_out["sales"]) - prediction)

    absolute_errors = np.abs(np.asarray(errors, dtype=float))
    return {
        "backtest_periods": len(errors),
        "backtest_mae": _stable_mean(absolute_errors),
        "backtest_rmse": _stable_rmse(errors),
    }


def summarize_forecast(history, predicted_sales):
    """Return transparent context and out-of-sample backtest metrics when possible."""
    history = validate_history(history)
    predicted_sales = float(_finite_predictions([predicted_sales])[0])
    latest_month = float(history["month"].max())
    latest_sales = float(history.loc[history["month"].idxmax(), "sales"])

    # These are in-sample fit diagnostics, not a guarantee of future forecast accuracy.
    fit_model = LinearRegression()
    centered_months = history[["month"]] - float(history["month"].min())
    fit_model.fit(centered_months, history["sales"])
    residuals = history["sales"].to_numpy(dtype=float) - fit_model.predict(centered_months)
    fit_mae = _stable_mean(np.abs(residuals))
    fit_rmse = _stable_rmse(residuals)

    if predicted_sales > latest_sales:
        trend = "increasing"
    elif predicted_sales < latest_sales:
        trend = "decreasing"
    else:
        trend = "stable"
    summary = {
        "next_month": latest_month + 1,
        "historical_average": _stable_mean(history["sales"].to_numpy(dtype=float)),
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
