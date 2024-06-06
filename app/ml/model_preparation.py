import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_log_error
import joblib
import os


# Определяем гиперпараметры


def model_preparation():
    # загружаем выборки
    X_train = pd.read_csv(os.path.join(os.path.dirname(__file__), "train/X_train.csv"))
    y_train = pd.read_csv(os.path.join(os.path.dirname(__file__), "train/y_train.csv")).squeeze()
    X_valid = pd.read_csv(os.path.join(os.path.dirname(__file__), "train/X_train.csv"))
    y_valid_log = pd.read_csv(os.path.join(os.path.dirname(__file__), "train/y_train.csv")).squeeze()
    
    #
    model_GB = GradientBoostingRegressor(
    learning_rate=0.5,
    n_estimators=100,
    max_depth=6,
    min_samples_split=30,
    random_state=42,
    verbose=1,
    )
    
    model_GB.fit(X_train, y_train)

    y_train_pred_GB = model_GB.predict(X_train)
    y_valid_pred_GB = model_GB.predict(X_valid)
    joblib.dump(model_GB, os.path.join(os.path.dirname(__file__), "model/model_GB.pkl"))
    
    rmse_gb_train = root_mean_squared_log_error(y_train, y_train_pred_GB)
    print(f'ошибка на тренировочной {rmse_gb_train:.2f}')
    rmse_gb_valid = root_mean_squared_log_error(y_valid_log, y_valid_pred_GB)
    print(f'ошибка на валидационной {rmse_gb_valid:.2f}')


model_preparation()