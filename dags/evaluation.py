import logging
from pathlib import Path, PosixPath
import pandas as pd
from datetime import date
from joblib import load
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from training import MODEL_PATH, get_data, give_ytrain_ytest
from sklearn.preprocessing import MinMaxScaler

def evaluate():
    df = get_data()
    m = give_ytrain_ytest()
    model_path = Path(MODEL_PATH)
    model = load_model(model_path.joinpath(f'model_{date.today().isoformat()}.joblib'))
    evaluate_model(model, df, m)

def load_model(model_path: PosixPath) -> Pipeline:
    return load(model_path)

def evaluate_model(model: Pipeline, df: pd.DataFrame, m: list):
    scaler = MinMaxScaler(feature_range=(0, 1))
    predictions = model.predict(df)
    train_predict = scaler.inverse_transform(predictions)
    
    logging.info(f'>>>> Mean squared error: {mean_squared_error(m, train_predict)}')
