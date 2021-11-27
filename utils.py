#!/usr/bin/python

import numpy
import numpy as np
import pandas

TICKER = 'Ticker'
OPEN = "Open"
HIGH = "High"
LOW = "Low"
CLOSE = "Close"
ADJ_CLOSE = "Adj Close"
VOLUME = "Volume"

TMP_DATASET = {}
DATA_PATH = "data_path"
KNOWN_DAYS = "known_days"
PREDICTED_DAYS = "predicted_days"


def denormalize(data_array, mean, std):
    return (data_array * std) + mean


def normalize(data_array):
    std = numpy.std(data_array)
    mean = numpy.mean(data_array)
    return ((data_array - mean) / std), mean, std


def normalize_data(data):
    new_data = []
    means = []
    stds = []
    for row in data:
        arr, mean, std = normalize(row[[OPEN, HIGH, LOW, CLOSE]].to_numpy())
        row[[OPEN, HIGH, LOW, CLOSE]] = arr
        row[[VOLUME]] = normalize(row[[VOLUME]].to_numpy())[0]
        new_data.append(row)
        means.append(mean)
        stds.append(std)
    return new_data, means, stds


def read_data(path):
    # type: (str) -> (list[pandas.DataFrame],list)
    df = pandas.read_csv(path)
    list_of_ticker = df[TICKER]
    tickers = set(list_of_ticker)
    print(f"Tickers: {len(tickers)}")
    data = [(df[df[TICKER] == ticker]).drop(columns=[TICKER, "Date", ADJ_CLOSE]).reset_index(drop=True) for
            ticker in tickers]
    return data, tickers


def make_dataset(raw_data, known_days, predicted_days):
    # type: (list[pandas.Dataframe], int, int) -> list
    def count_y(base_data, frame):
        # type: (float, pandas.DataFrame) -> list
        tmp_data = [(price, (price / base_data) - 1) for price in frame[CLOSE]]
        sorted_data = sorted(tmp_data, key=lambda x: abs(x[1]), reverse=True)
        return [sorted_data[0][0]]

    def convert_to_data_set(data_frames, index):
        # type: (list[pandas.DataFrame], int) -> list
        tmp_data_set = []
        total = len(data_frames)
        i = 1
        for data_frame in data_frames:
            while not data_frame.empty:
                raw_x = data_frame[0:known_days].reset_index(drop=True)
                if len(data_frame.index) < predicted_days + known_days:
                    break
                y = count_y(list(raw_x[CLOSE])[-1],
                            data_frame[known_days:known_days + predicted_days].reset_index(drop=True))
                tmp_data_set.append([raw_x.to_numpy(), y])
                data_frame = data_frame.drop([0]).reset_index(drop=True)
            print(f'[Thread: {index}] {i}/{total}')
            i += 1

        TMP_DATASET[index] = tmp_data_set
        return tmp_data_set

    return convert_to_data_set(raw_data, 0)


def remove_ndarray(data):
    return [[np.array(row[0]).tolist(), row[1]] for row in data]


def get_verification_dataset(datapath, known_days, predicted_days):
    ver_data = {}
    raw_data, tickers = read_data(datapath)
    print(len(raw_data[0]))
    print(tickers)
    print("ver data is read")
    tickers = list(tickers)
    for i in range(len(tickers)):
        ver_data[tickers[i]] = raw_data[i]
    print("ver data is formed")
    for key, value in ver_data.items():
        data1, mean, std = normalize_data([value])
        ver_data[key] = (make_dataset(data1, known_days, predicted_days), mean[0], std[0])
    return ver_data


def get_dataset(datapath, known_days, predicted_days):
    print(f"Read data from: {datapath}")
    raw_data = read_data(datapath)[0]
    print("Data is read")
    data = normalize_data(raw_data)[0]
    print("dataset is normalized")
    print(f"Create dataset {KNOWN_DAYS} = {known_days}, {PREDICTED_DAYS} = {predicted_days}")
    data = make_dataset(data, known_days, predicted_days)
    print("Dataset is created")

    return remove_ndarray(data)
