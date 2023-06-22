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

wmt = yf.Ticker('WMT')
d1 = wmt.history(start="2023-02-01",end="2023-06-23",interval='1d')
print(type(d1))
print(d1)
#print(data,dir(data))
exit(0)

# convert to numpy
data_numpy = data.values

# convert to PyTorch tensor
data_tensor = torch.from_numpy(data_numpy)

# Print
print(data_tensor)
