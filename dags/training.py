import logging
from pathlib import Path, PosixPath
import numpy as np
import pandas as pd
from datetime import date
from joblib import dump
import pandas_datareader as pdr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

MODEL_PATH = 'C:/airflow-2023/model'

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def training():
    df = get_data()
    model = build_ml_pipeline(df)
    model.fit(df, epochs=100, batch_size=70)
    save_model(model, Path(MODEL_PATH))

def get_data():
    key = "xxxxxxxxxxxxxxx"  # your tiingo api key
    df = pdr.get_data_tiingo('NDAQ', api_key=key)
    dirname = 'C:/NASDAQ-model/model/NDAQ2.csv'
    df.to_csv(dirname)
    df = pd.read_csv('C:/NASDAQ-model/model/NDAQ2.csv')
    df1 = df.reset_index()['close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    training_size = int(len(df1) * 0.70)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, y_train

def build_ml_pipeline(df: pd.DataFrame) -> Pipeline:
    model = Sequential()
    model.add(LSTM(60, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(60, return_sequences=True))
    model.add(LSTM(60))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    logging.info(f'### A number of numerical features: {model.summary}')
    return model

def save_model(model: Pipeline, model_path: PosixPath):
    path = model_path.joinpath(f'model_{date.today().isoformat()}.joblib')
    dump(model, path)

def give_ytrain_ytest():
    df = pd.read_csv('C:/NASDAQ-model/model/NDAQ2.csv')
    df1 = df.reset_index()['close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    training_size = int(len(df1) * 0.70)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    return y_train
