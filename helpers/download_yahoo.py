#!/usr/bin/python
# Import package
import os
from datetime import datetime, timedelta

import yfinance as yf

current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
save_dir = root_dir + "/resources/yahoo"

tikers_rus = ['GAZP.ME', 'SBER.ME', 'SBERP.ME', 'LKOH.ME', 'GMKN.ME', 'VTBR.ME', 'PLZL.ME', 'TATN.ME', 'ROSN.ME',
              'MGNT.ME', 'RTKM.ME', 'RTKMP.ME', 'MTSS.ME', 'SNGS.ME', 'SNGSP.ME', 'CHMF.ME', 'MOEX.ME', 'AFLT.ME',
              'YNDX.ME', 'TCSG.ME', 'LSRG.ME', 'PIKK.ME', 'AFKS.ME', 'ALRS.ME', 'POLY.ME', 'IRAO.ME',
              'MAGN.ME', 'DSKY.ME', 'FEES.ME']
tikers_us = ['MMM', 'SPX', 'ABBV', 'ADBE', 'AA', 'AKAM', 'GOOG', 'AMZN', 'AXP', 'AAPL', 'APA', 'AMAT', 'ADSK',
             'T', 'BBBY', 'BMY', 'CF', 'CSCO', 'C', 'CCE', 'CVS', 'COP', 'CMI', 'DHR', 'DRI', 'DAL', 'EBAY', 'FDX',
             'FE', 'F', 'FB', 'MAR', 'GE', 'GM', 'HPQ', 'INTC', 'JNJ', 'M', 'MA', 'MCD', 'MRK', 'MU', 'MSFT',
             'MOS', 'NKE', 'NVDA', 'PYPL', 'PEP', 'PFE', 'QRVO', 'QCOM', 'LUV', 'VZ', 'V', 'XLNX', 'NTAP', 'NEM',
             'NEE', 'YUM', 'XOM', 'PLUG']
tikers = tikers_us + tikers_rus
# Get the dat
start_date = datetime(1990, 1, 1)
end_date = datetime(2021, 11, 18)
time_delta = timedelta(7)
current_date = start_date
str_start_date = current_date.strftime("%Y-%m-%d")
str_end_date = end_date.strftime("%Y-%m-%d")

for tiker in tikers:
    data = yf.download(tickers=tiker, start=str_start_date, end=str_end_date, interval="1d")
    data.to_csv(f'{save_dir}/{tiker}.csv')
