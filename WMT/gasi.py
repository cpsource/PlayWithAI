# where.from: https://medium.com/swlh/free-historical-market-data-download-in-python-74e8edd462cf
#
# See Also: https://algotrading101.com/learn/yfinance-guide/
#
#!pip install yfinance
#!pip install mplfinance
#
from datetime import datetime
import yfinance as yf
import mplfinance as mpf
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

file_path = 'wmt-1m.pkl'

if os.path.exists(file_path):
    print("Loading from cache")
    with open(file_path,"rb") as f:
        d1 = pickle.load(f)
else:
    print("Loading from Yahoo")
    wmt = yf.Ticker('WMT')
    d1 = wmt.history(start="2023-06-18",end="2023-06-23",interval='1m')
    with open(file_path,"wb") as f:
        pickle.dump(d1,f)

with pd.option_context("display.max_rows", None , "display.max_columns", None):
    print(type(d1))
    print(d1)
#print(dir(d1))
#print(d1.ndim)
#print(d1.shape)
#print(d1.dtypes)
#print(d1['Close'])

print(f"d1.keys() = {d1.keys}\nDone.")

exit(0)

# cponvert to numpy
data_numpy = d1.values

# convert to PyTorch tensor
data_tensor = torch.from_numpy(data_numpy)[:,3]
data_tensor = data_tensor[:388]
min_value = torch.min(data_tensor)
max_value = torch.max(data_tensor)
data_tensor = (data_tensor - min_value)/(max_value - min_value)
print(data_tensor.shape)

# Print
print(data_tensor.shape)

plt.plot(data_tensor)
plt.show()
