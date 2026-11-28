import zlib
import gzip
import bz2
import lzma
import io
import csv
import tempfile
import os
from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

REQUIRED_COLUMNS = {"month", "sales"}
MAX_SAFE_MONTH_INDEX = 2**53 - 1
MAX_FORECAST_PERIODS = 1200  # Bound memory use to at most 100 years of monthly output.


def _read_csv_input(csv_file, delimiter=",", max_rows=None, encoding="utf-8", **options):
    if encoding not in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        raise ValueError("Unsupported input encoding.")
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 10_000_000):
        raise ValueError("max_rows must be an integer from 1 to 10000000.")
    if delimiter not in (",", ";", "\t", "|"):
        raise ValueError("Unsupported input delimiter.")
    if max_rows is not None:
        if "nrows" in options:
            raise ValueError("Do not combine nrows and max_rows.")
        options["nrows"] = max_rows + 1
    _reject_duplicate_headers(csv_file, delimiter, encoding=encoding)
    try:
        frame = pd.read_csv(csv_file, **({"encoding": encoding} if encoding != "utf-8" else {}), **options, **({"sep": delimiter} if delimiter != "," else {}))
    except (EOFError, lzma.LZMAError, zlib.error, csv.Error) as exc:
        raise ValueError("CSV input is corrupt or incomplete, or its header exceeds parser limits.") from exc
    if max_rows is not None and len(frame) > max_rows:
        raise ValueError("CSV input exceeds the configured row limit.")
    return frame


def _reject_duplicate_headers(csv_file, delimiter=",", encoding="utf-8"):
    """Inspect original UTF-8 CSV headers without taking ownership of streams."""
    def header_lines(source):
        # csv.reader needs physical lines even for StringIO objects whose default
        # iterator does not recognize CR-only records. Preserve quoted newlines.
        pending = ""
        first = True
        started = False
        while True:
            chars = []
            while True:
                char = pending or source.read(1)
                pending = ""
                if not char:
                    break
                if first:
                    first = False
                    if char == "\ufeff":
                        continue
                chars.append(char)
                if char == "\r":
                    following = source.read(1)
                    if following == "\n":
                        chars.append(following)
                    else:
                        pending = following
                    break
                if char == "\n":
                    break
            if not chars:
                return
            line = "".join(chars)
            if not started and not line.strip():
                continue
            started = True
            yield line

    if isinstance(csv_file, (str, os.PathLike)):
        if not Path(csv_file).is_file():
            return
        opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(Path(csv_file).suffix, open)
        try:
            with opener(csv_file, mode="rt", encoding=encoding, newline="") as source:
                headers = next(csv.reader(header_lines(source), delimiter=delimiter), [])
        except (EOFError, lzma.LZMAError, zlib.error, csv.Error) as exc:
            raise ValueError("CSV input is corrupt or incomplete, or its header exceeds parser limits.") from exc
    else:
        if not all(hasattr(csv_file, name) for name in ("read", "seek", "tell")) or (hasattr(csv_file, "seekable") and not csv_file.seekable()):
            raise ValueError("CSV streams must be seekable for header validation.")
        position = csv_file.tell()
        binary = isinstance(csv_file.read(0), bytes)
        source = io.TextIOWrapper(csv_file, encoding=encoding, newline="") if binary else csv_file
        try:
            headers = next(csv.reader(header_lines(source), delimiter=delimiter), [])
        except (EOFError, lzma.LZMAError, zlib.error, csv.Error) as exc:
            raise ValueError("CSV input is corrupt or incomplete, or its header exceeds parser limits.") from exc
        finally:
            if binary:
                source.detach()
            csv_file.seek(position)
    if len(headers) != len(set(headers)):
        raise ValueError("CSV column names must be unique; duplicate headers are ambiguous.")


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
    try:
        return math.fsum(float(value) for value in values) / len(values)
    except OverflowError:
        from fractions import Fraction
        return float(sum((Fraction.from_float(float(value)) for value in values), Fraction()) / len(values))


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
    if any(pd.api.types.is_complex_dtype(ordered[column]) for column in ordered):
        raise ValueError("History must contain real numbers, not complex values.")
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


def load_sales_history(csv_file, delimiter=",", max_rows=None, encoding="utf-8"):
    """Load and validate monthly sales data, returning it in month order."""
    frame = _read_csv_input(csv_file, delimiter, **({"encoding": encoding} if encoding != "utf-8" else {}), **({"max_rows": max_rows} if max_rows is not None else {}))
    return validate_history(frame)


def _fit_scaled_trend(model, months, sales):
    scale = float(np.max(np.abs(sales))) or 1.0
    model.fit(months, sales / scale)
    model._forecast_target_scale = scale


def _predict_scaled_trend(model, months):
    return _finite_predictions(model.predict(months) * model._forecast_target_scale)


def predict_next_sales(df, non_negative=False):
    """Fit a linear trend and predict the period after the latest observed month."""
    if not isinstance(non_negative, bool):
        raise ValueError("non_negative must be a boolean.")
    ordered = validate_history(df)
    model = LinearRegression()
    origin = float(ordered["month"].iloc[0])
    _fit_scaled_trend(model, ordered[["month"]] - origin, ordered["sales"])
    if float(ordered["month"].iloc[-1]) >= MAX_SAFE_MONTH_INDEX:
        raise ValueError("Next month exceeds exact floating-point integer precision.")
    next_month = float(ordered["month"].iloc[-1]) + 1
    value = float(_finite_predictions(_predict_scaled_trend(model, pd.DataFrame({"month": [next_month - origin]})))[0])
    return max(0.0, value) if non_negative else value


def predict_sales_horizon(history, periods=1, non_negative=False):
    """Predict one or more future monthly sales values from the fitted trend."""
    if not isinstance(non_negative, bool):
        raise ValueError("non_negative must be a boolean.")
    if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
        raise ValueError("periods must be a positive integer.")
    if periods > MAX_FORECAST_PERIODS:
        raise ValueError(f"Forecast horizon cannot exceed {MAX_FORECAST_PERIODS} months.")
    ordered = validate_history(history)
    model = LinearRegression()
    origin = float(ordered["month"].iloc[0])
    _fit_scaled_trend(model, ordered[["month"]] - origin, ordered["sales"])
    if int(ordered["month"].iloc[-1]) + periods > MAX_SAFE_MONTH_INDEX:
        raise ValueError("Forecast horizon exceeds exact floating-point integer precision.")
    first_future_month = int(ordered["month"].iloc[-1]) + 1
    future_months = np.array([first_future_month + offset for offset in range(periods)], dtype=float)
    predictions = _finite_predictions(_predict_scaled_trend(model, pd.DataFrame({"month": future_months - origin})))
    if non_negative:
        predictions = np.maximum(predictions, 0)
    return pd.DataFrame({"month": future_months, "predicted_sales": predictions.astype(float)})


def backtest_forecast(history, max_periods=1000, non_negative=False):
    """Evaluate one-period-ahead predictions using expanding historical windows.

    At least three observations are required so that the first evaluation has
    two training periods and one held-out period. The function never trains on
    the observation it evaluates. Validate inputs here as callers may invoke
    backtesting directly without going through load_sales_history().
    """
    if not isinstance(non_negative, bool):
        raise ValueError("non_negative must be a boolean.")
    if not isinstance(max_periods, int) or isinstance(max_periods, bool) or max_periods < 1:
        raise ValueError("max_periods must be a positive integer.")
    ordered = validate_history(history, minimum_rows=1)
    if len(ordered) < 3:
        return None

    errors = []
    naive_errors = []
    naive_scale = float(np.max(np.abs(ordered["sales"]))) or 1.0
    first_split = max(2, len(ordered) - max_periods)
    for split_index in range(first_split, len(ordered)):
        training = ordered.iloc[:split_index]
        held_out = ordered.iloc[split_index]
        model = LinearRegression()
        origin = float(training["month"].iloc[0])
        _fit_scaled_trend(model, training[["month"]] - origin, training["sales"])
        prediction = float(
            _finite_predictions(_predict_scaled_trend(model, pd.DataFrame({"month": [float(held_out["month"]) - origin]})))[0]
        )
        if non_negative:
            prediction = max(0.0, prediction)
        errors.append(float(held_out["sales"]) - prediction)
        naive_errors.append(float(held_out["sales"]) / naive_scale - float(training["sales"].iloc[-1]) / naive_scale)

    absolute_errors = np.abs(np.asarray(errors, dtype=float))
    naive_mae = _stable_mean(np.abs(naive_errors)) * naive_scale
    naive_rmse = _stable_rmse(naive_errors) * naive_scale
    naive_mae = naive_mae if math.isfinite(naive_mae) else None
    naive_rmse = naive_rmse if math.isfinite(naive_rmse) else None
    skill = 100 * (1 - _stable_mean(absolute_errors) / naive_mae) if naive_mae is not None and naive_mae > 0 else None
    skill = skill if skill is not None and math.isfinite(skill) else None
    return {
        "backtest_periods": len(errors),
        "backtest_available_periods": len(ordered) - 2,
        "backtest_start_month": float(ordered["month"].iloc[first_split]),
        "backtest_mae": _stable_mean(absolute_errors),
        "backtest_naive_mae": naive_mae,
        "backtest_naive_rmse": naive_rmse,
        "backtest_mae_skill_percent": skill,
        "backtest_rmse": _stable_rmse(errors),
    }


def summarize_forecast(history, predicted_sales, backtest_periods=1000, non_negative=False):
    """Return transparent context and out-of-sample backtest metrics when possible."""
    if not isinstance(non_negative, bool):
        raise ValueError("non_negative must be a boolean.")
    history = validate_history(history)
    predicted_sales = float(_finite_predictions([predicted_sales])[0])
    if non_negative:
        predicted_sales = max(0.0, predicted_sales)
    latest_month = int(history["month"].max())
    if latest_month >= MAX_SAFE_MONTH_INDEX:
        raise ValueError("Next month exceeds exact floating-point integer precision.")
    latest_sales = float(history.loc[history["month"].idxmax(), "sales"])

    # These are in-sample fit diagnostics, not a guarantee of future forecast accuracy.
    fit_model = LinearRegression()
    centered_months = history[["month"]] - float(history["month"].min())
    _fit_scaled_trend(fit_model, centered_months, history["sales"])
    residuals = history["sales"].to_numpy(dtype=float) - _predict_scaled_trend(fit_model, centered_months)
    fit_mae = _stable_mean(np.abs(residuals))
    fit_rmse = _stable_rmse(residuals)

    if predicted_sales > latest_sales:
        trend = "increasing"
    elif predicted_sales < latest_sales:
        trend = "decreasing"
    else:
        trend = "stable"
    period_span = int(history["month"].max()) - int(history["month"].min()) + 1
    summary = {
        "next_month": latest_month + 1,
        "history_period_span": period_span,
        "missing_history_periods": period_span - len(history),
        "historical_average": _stable_mean(history["sales"].to_numpy(dtype=float)),
        "predicted_sales": float(predicted_sales),
        "trend": trend,
        "historical_fit_mae": fit_mae,
        "historical_fit_rmse": fit_rmse,
    }
    if non_negative:
        summary["forecast_policy"] = "non_negative"
    backtest = backtest_forecast(history, **({"non_negative": True} if non_negative else {}), **({"max_periods": backtest_periods} if backtest_periods != 1000 else {}))
    if backtest is not None:
        summary.update(backtest)
    return summary


def select_history_window(history, lookback_periods=None):
    history = validate_history(history)
    available = len(history)
    if lookback_periods is not None:
        if not isinstance(lookback_periods, int) or isinstance(lookback_periods, bool) or lookback_periods < 2:
            raise ValueError("lookback_periods must be an integer of at least two.")
        history = history.tail(min(lookback_periods, available)).reset_index(drop=True)
    return history, available


def forecast_sales(csv_file, horizon=1, delimiter=",", lookback_periods=None, backtest_periods=1000, max_rows=None, non_negative=False, encoding="utf-8"):
    if not isinstance(non_negative, bool):
        raise ValueError("non_negative must be a boolean.")
    try:
        history = load_sales_history(csv_file, **({"encoding": encoding} if encoding != "utf-8" else {}), **({"max_rows": max_rows} if max_rows is not None else {}), **({"delimiter": delimiter} if delimiter != "," else {}))
        history, available = select_history_window(history, lookback_periods)
        forecast = predict_sales_horizon(history, periods=horizon, **({"non_negative": True} if non_negative else {}))
        predicted_sales = float(forecast["predicted_sales"].iloc[0])
        print("\n📈 Sales Forecasting AI")
        print("-----------------------")
        print("Training model using historical data...")
        if len(history) < available:
            print(f"Using latest {len(history)} of {available} observed history periods.")
        summary = summarize_forecast(history, predicted_sales, **({"non_negative": True} if non_negative else {}), **({"backtest_periods": backtest_periods} if backtest_periods != 1000 else {}))
        print(f"Predicted sales for next period: ₹{predicted_sales:.2f}")
        print(f"Historical average: ₹{summary['historical_average']:.2f}")
        print(f"Trend: {summary['trend']}")
        print(f"Historical fit MAE (in-sample): ₹{summary['historical_fit_mae']:.2f}")
        print(f"Historical fit RMSE (in-sample): ₹{summary['historical_fit_rmse']:.2f}")
        if "backtest_mae" in summary:
            if summary["backtest_periods"] < summary["backtest_available_periods"]:
                print(f"Backtest covers the latest {summary['backtest_periods']} of {summary['backtest_available_periods']} eligible periods.")
            print(
                f"Rolling backtest MAE ({summary['backtest_periods']} periods): "
                f"₹{summary['backtest_mae']:.2f}"
            )
            print(f"Rolling backtest RMSE: ₹{summary['backtest_rmse']:.2f}\n")
            if summary["backtest_naive_mae"] is not None:
                print(f"Last-observation baseline backtest MAE: ₹{summary['backtest_naive_mae']:.2f}")
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


def export_forecast_csv(frame, output_path, source_path):
    output_path = Path(output_path).expanduser().resolve()
    source_path = Path(source_path).expanduser().resolve()
    if output_path == source_path or (output_path.exists() and output_path.samefile(source_path)):
        raise ValueError("Forecast export must differ from the source history")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Forecast monthly sales from a CSV history.")
    parser.add_argument("csv_file", help="CSV containing month and sales columns")
    parser.add_argument("--horizon", type=int, default=1, help="Number of future months to forecast")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable forecast and diagnostic summary")
    parser.add_argument("--output", help="Atomically export every forecast period to a CSV different from the input")
    parser.add_argument("--delimiter", choices=(",", ";", "tab", "|"), default=",", help="Input CSV delimiter; exports remain comma-separated")
    parser.add_argument("--lookback-periods", type=int, help="Fit and diagnose only the latest N observed periods")
    parser.add_argument("--backtest-periods", type=int, default=1000, help="Maximum recent held-out periods evaluated by rolling backtesting")
    parser.add_argument("--max-input-rows", type=int, help="Reject CSV files exceeding this raw row budget (1..10000000)")
    parser.add_argument("--non-negative", action="store_true", help="Clamp future and held-out trend predictions to zero")
    parser.add_argument("--encoding", choices=("utf-8", "utf-8-sig", "cp1252", "latin-1"), default="utf-8", help="Input CSV text encoding; output encoding remains UTF-8")
    args = parser.parse_args(argv)
    if args.max_input_rows is not None and not 1 <= args.max_input_rows <= 10_000_000:
        parser.error("--max-input-rows must be from 1 to 10000000")
    if args.backtest_periods < 1:
        parser.error("--backtest-periods must be positive")
    delimiter = "\t" if args.delimiter == "tab" else args.delimiter
    if args.json or args.output:
        import json
        try:
            history = load_sales_history(args.csv_file, **({"encoding": args.encoding} if args.encoding != "utf-8" else {}), **({"max_rows": args.max_input_rows} if args.max_input_rows is not None else {}), **({"delimiter": delimiter} if delimiter != "," else {}))
            history, available = select_history_window(history, args.lookback_periods)
            result = predict_sales_horizon(history, args.horizon, **({"non_negative": True} if args.non_negative else {}))
            payload = {"forecast": result.to_dict(orient="records"), "summary": summarize_forecast(history, result["predicted_sales"].iloc[0], **({"non_negative": True} if args.non_negative else {}), **({"backtest_periods": args.backtest_periods} if args.backtest_periods != 1000 else {}))}
            payload["summary"]["training_periods"] = len(history)
            payload["summary"]["available_history_periods"] = available
            if args.output:
                export_forecast_csv(result, args.output, args.csv_file)
            if args.json:
                print(json.dumps(payload, allow_nan=False))
            else:
                print(result.to_string(index=False))
            return 0
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            parser.exit(1, f"Error: {exc}\n")
    return 0 if forecast_sales(args.csv_file, **({"encoding": args.encoding} if args.encoding != "utf-8" else {}), horizon=args.horizon, **({"non_negative": True} if args.non_negative else {}), **({"max_rows": args.max_input_rows} if args.max_input_rows is not None else {}), **({"delimiter": delimiter} if delimiter != "," else {}), **({"lookback_periods": args.lookback_periods} if args.lookback_periods is not None else {}), **({"backtest_periods": args.backtest_periods} if args.backtest_periods != 1000 else {})) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
