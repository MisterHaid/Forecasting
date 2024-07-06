import pandas as pd
from prophet.serialize import model_to_json, model_from_json
class ProphetModel:
    def __init__(self, model_path='final_model.json'):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        with open('final_model.json', 'r') as fin:
            model = model_from_json(fin.read())  # Load model
        return model

    def predict(self, df, period):
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=period)

        # Make forecast
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat']].tail(period)
