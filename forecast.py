import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

def forecast_sales(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if not {"month", "sales"}.issubset(df.columns):
            print("❌ CSV must contain 'month' and 'sales' columns.")
            return
        
        df = df.copy()
        for column in ("month", "sales"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if df.empty or df[["month", "sales"]].isna().any().any():
            print("❌ Sales history must contain numeric month and sales values.")
            return
        if not all(pd.notna(value) and abs(float(value)) != float("inf") for value in df[["month", "sales"]].to_numpy().ravel()):
            print("❌ Month and sales values must be finite numbers.")
            return
        if df["month"].nunique() < 2:
            print("❌ At least two distinct months are required for a forecast.")
            return
        df = df.sort_values("month")
        X = df[["month"]]
        y = df["sales"]

        model = LinearRegression()
        model.fit(X, y)

        next_month = [[X.iloc[-1][0] + 1]]
        predicted_sales = model.predict(next_month)[0]

        print("\n📈 Sales Forecasting AI")
        print("-----------------------")
        print("Training model using historical data...")
        print(f"Predicted sales for next period: ₹{predicted_sales:.2f}\n")

    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python forecast.py <csv_file>")
        return

    forecast_sales(sys.argv[1])

if __name__ == "__main__":
    main()
