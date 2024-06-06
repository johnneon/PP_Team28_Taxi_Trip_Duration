import joblib
import os
import numpy as np
import pandas as pd


def predict(data):
    """ Travel time prediction """

    # Load model

    with open(os.path.join(os.path.dirname(__file__), "../ml/model/model_GB.pkl"), 'rb') as file:
        model = joblib.load(file)
        prediction = model.predict(data)
        prediction = pd.DataFrame(prediction, columns=['trip_duration_log_pred_gb'])
        predict_normal = np.exp(prediction['trip_duration_log_pred_gb'])-1
        result = pd.DataFrame({'trip_duration': predict_normal})
        return result