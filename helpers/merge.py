#!/usr/bin/python

import os

import pandas
from pandas import DataFrame

TICKER = "Ticker"
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
files_dir = root_dir + "/resources/yahoo"
save_dir = root_dir + "/resources/results"
RESULT_FILE = f'{save_dir}/yahoo_r_result.csv'

EXCLUDE = ["AAPL", "MSFT"]
NEED = ['GAZP.ME', 'SBER.ME', 'LKOH.ME', 'GMKN.ME', 'VTBR.ME', 'PLZL.ME', 'ROSN.ME', 'AFLT.ME',
        'AA', 'GOOG', 'AMZN', 'CF', 'CSCO', 'C', 'CMI', 'DRI', 'DAL', 'EBAY', 'FDX',
        'GE', 'GM', 'HPQ', 'INTC', 'JNJ', 'M', 'MA']

files = os.listdir(files_dir)
print(len(files))
data = {}
i = 0
for file in files:
    tiker = file[:-4]
    if tiker in EXCLUDE:
        continue
    if tiker not in NEED:
        continue
    df = pandas.read_csv(f'{files_dir}/{file}')
    data[tiker] = df
    i += 1

example = list(data.keys())[0]
df = data[example]
df = DataFrame(columns=[TICKER] + list(df.keys()))
rows = 0
for ticker, prices in data.items():
    tickers = [ticker for _ in range(len(prices.index))]
    rows += len(tickers)
    prices[TICKER] = tickers
print(rows)
df = pandas.concat(data.values(), ignore_index=True)

df.to_csv(path_or_buf=RESULT_FILE, index=False)
