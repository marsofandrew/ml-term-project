#!/usr/bin/python
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import models

from utils import get_verification_dataset, denormalize, read_data, CLOSE

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
RESOURCE_DIR = f'{ROOT_DIR}/resources'
SOURCE_PATH = "source_path"

KNOWN_DAYS = "known_days"
PREDICTED_DAYS = "predicted_days"
BATCH = "batch"
VERIFICATION_DATA_PATH = "verification_data_path"
BASE_DIR = "base_dir"

CONFIG = {
    KNOWN_DAYS: 60,
    PREDICTED_DAYS: 1,
    BATCH: 1,
    VERIFICATION_DATA_PATH: f'{RESOURCE_DIR}/results/verification.csv',
    BASE_DIR: f"{RESOURCE_DIR}/model"
}


def plot_data(ticker, y, predicted_y, mean, std, raw_y):
    denormalized_y = denormalize(np.array(y), mean, std)
    denormalized_pred = denormalize(np.array(predicted_y), mean, std)
    y_list = np.array(denormalized_y).flatten().tolist()
    predicted_list = np.array(denormalized_pred).flatten().tolist()
    plt.plot(y_list, color="b", label='real data')
    plt.plot(predicted_list, color='green', label="predicted data")
    plt.plot(raw_y, 'k', label="prices")
    plt.legend()
    plt.title(ticker)
    plt.show()


raw_data, tickers = read_data(CONFIG[VERIFICATION_DATA_PATH])
tickers = list(tickers)
source_data = {}
for i in range(len(raw_data)):
    source_data[tickers[i]] = raw_data[i][CONFIG[KNOWN_DAYS]-CONFIG[PREDICTED_DAYS]:]

verification_data_dict = get_verification_dataset(CONFIG[VERIFICATION_DATA_PATH], CONFIG[KNOWN_DAYS],
                                                  CONFIG[PREDICTED_DAYS])

model_path = f"{CONFIG[BASE_DIR]}/{sys.argv[1]}_model.h5"

model = models.load_model(model_path)
print(model.summary())

for ticker, value in verification_data_dict.items():
    data = value[0]
    mean = value[1]
    std = value[2]
    print(mean, std)
    y = np.array(data)[:, 1].tolist()
    x = [ell[0].tolist() for ell in data]
    loss, error = model.evaluate(x, y, batch_size=CONFIG[BATCH])
    print(f'Ticker: {ticker}; accuracy = {100 - error}')
    predicted = model.predict(x, batch_size=CONFIG[BATCH])
    plot_data(ticker, y, predicted, mean, std, source_data[ticker][CLOSE].to_numpy().tolist())
