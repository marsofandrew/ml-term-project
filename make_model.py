#!/usr/bin/python

import os
import sys

import numpy as np
from tensorflow.keras import layers as tfl
from tensorflow.keras import losses as tf_loss
from tensorflow.keras.models import Sequential
from utils import get_dataset

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
RESOURCE_DIR = f'{ROOT_DIR}/resources'
SOURCE_PATH = "source_path"

DATA_PATH = "data_path"
KNOWN_DAYS = "known_days"
PREDICTED_DAYS = "predicted_days"
THREADS_TO_CREATE = "threads_to_create"
PREFIX = "prefix"
BATCH = "batch"
VERIFICATION_DATA_PATH = "verification_data_path"
MODEL_SAVE_DIR = "model_save_path"

CONFIG = {
    DATA_PATH: f'{RESOURCE_DIR}/results/yahoo_r_result.csv',
    KNOWN_DAYS: 60,
    PREDICTED_DAYS: 1,
    THREADS_TO_CREATE: 1,
    PREFIX: "test",
    BATCH: 8,
    VERIFICATION_DATA_PATH: f'{RESOURCE_DIR}/results/verification.csv',
    MODEL_SAVE_DIR: f'{RESOURCE_DIR}/model'
}

TYPE = "full_p1"


def _split_data(data_frame, divide):
    # type: (list, int) -> (np.array, np.array)
    total_amount = len(data_frame)
    verification_amount = total_amount // divide + 1
    new_list = list(data_frame)
    return np.array(new_list[total_amount-verification_amount:total_amount]), np.array(new_list[0:total_amount-verification_amount])


def split_to_train_verification(data_frame):
    # type: (list) -> (np.array, np.array)
    return _split_data(data_frame, 10)


def split_to_test_train(data_frame):
    # type: (list) -> (list, list)
    return _split_data(data_frame, 5)


def create_model(layers, loss_fn=tf_loss.MeanSquaredError()):
    model_layer = [tfl.Input(shape=(CONFIG[KNOWN_DAYS], 5))] + layers + [tfl.Dense(1)]
    model = Sequential(model_layer)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['mean_absolute_percentage_error'])
    return model


def train_model(layers, loss_fn, epoch, ttrain_data, test_data):
    model = create_model(layers, loss_fn=loss_fn)
    train_data = ttrain_data

    train_x = train_data[:, 0].tolist()
    train_y = train_data[:, 1].tolist()
    model.fit(train_x, train_y, epochs=epoch, batch_size=CONFIG[BATCH])
    predicted = model.predict(test_data[:, 0].tolist(), batch_size=CONFIG[BATCH])
    print(predicted)
    print(test_data[:, 1].tolist())
    #print(predicted[0], test_data[:, 1].tolist()[0])
    loss, error = model.evaluate(test_data[:, 0].tolist(), test_data[:, 1].tolist(), batch_size=CONFIG[BATCH])
    print(f'loss = {loss}, error = {error}')
    return model, error


data = get_dataset(CONFIG[DATA_PATH], CONFIG[KNOWN_DAYS], CONFIG[PREDICTED_DAYS])
test_data = np.array(get_dataset(CONFIG[VERIFICATION_DATA_PATH], CONFIG[KNOWN_DAYS], CONFIG[PREDICTED_DAYS]))
print("Data is read")
ttrain_data = np.array(data)
#verification_data_dict = get_verification_dataset(CONFIG[VERIFICATION_DATA_PATH], CONFIG[KNOWN_DAYS], CONFIG[PREDICTED_DAYS])
print("Data is split to verification and test")

variants = [
    #["t0", [tfl.LSTM(60), tfl.Dense(4)]], #Поиграться с эпохами 20, 40
    #["td0", [tfl.LSTM(60), tfl.Dropout(0.2), tfl.Dense(4), tfl.Activation('sigmoid')]], #Поиграться с эпохами 20, 40
    #["td21", [tfl.LSTM(60), tfl.Dropout(0.2), tfl.Dense(64)]],  #Поиграться с эпохами 20, 40, 50
    ["td22",
    [tfl.LSTM(60), tfl.Dropout(0.2), tfl.Dense(64), tfl.Dropout(0.2), tfl.Dense(4), tfl.Activation('sigmoid')]], # increase epochs 30+
    #["t2", [tfl.LSTM(60, activation='relu'), tfl.Dense(64, activation='relu')]], #10
    #["td2", [tfl.LSTM(60, activation='relu'), tfl.Dropout(0.2), tfl.Dense(64, activation='relu')]], #20
    #["td3", [tfl.LSTM(60, activation='relu'), tfl.Dropout(0.2), tfl.Dense(64, activation='relu'), tfl.Dense(4, activation='relu')]] #10
]
variants_res = []
print(sys.argv)
epochs = [int(a) for a in sys.argv[1:]]

print(epochs)
for variant in variants:
    for epoch in epochs:
        accuracies = []
        for _ in range(1):
            model, error = train_model(variant[1], tf_loss.MeanAbsolutePercentageError(), epoch, ttrain_data, test_data)
            model.save(f'{CONFIG[MODEL_SAVE_DIR]}/{variant[0]}_{TYPE}_{epoch}_model.h5')
            accuracies.append((100 - error))
        accuracy = np.mean(accuracies)
        variants_res.append((accuracy, epoch, variant[0]))
variants_res = sorted(variants_res, key=lambda x: x[0], reverse=True)

for variant in variants_res:
    print(f'variant epoch = {variant[1]}; accuracy = {variant[0]}; '
          f'variant = {variant[2]}')

