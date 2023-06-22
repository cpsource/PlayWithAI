# where.from: https://medium.com/swlh/free-historical-market-data-download-in-python-74e8edd462cf
#
# See Also: https://algotrading101.com/learn/yfinance-guide/
# See Also: https://medium.com/swlh/free-historical-market-data-download-in-python-74e8edd462cf
#
#!pip install yfinance
#!pip install mplfinance
#
from datetime import datetime
import yfinance as yf
import mplfinance as mpf
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 6, 22)
data = yf.download('WMT', start=start_date, end=end_date)
#print(data)
#mpf.plot(data,type='candle',mav=(3,6,9),volume=True,show_nontrading=True)

# convert to numpy
import numpy as np
data_numpy = data.values

# convert to PyTorch tensor
import torch
data_tensor = torch.from_numpy(data_numpy)

# Print
print(data_tensor)
