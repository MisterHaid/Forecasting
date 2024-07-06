import sys
import pandas as pd
from predict import ProphetModel

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_script.py <data_path> <period>")
        sys.exit(1)

    data_path = sys.argv[1]
    period = int(sys.argv[2])

    # Load data
    df = pd.read_csv(data_path)
    # Initialize ProphetModel with default model path
    model = ProphetModel()

    # Make prediction
    forecast = model.predict(df, period)
    forecast = forecast.join(df['y'], how='left')
    forecast.to_csv('result_of_forecast.csv')
    # Print forecast
    print(forecast)