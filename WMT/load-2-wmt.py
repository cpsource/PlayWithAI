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
from numpy.fft import fft, ifft

import matplotlib.pyplot as plt

def fourier_transform(x):
    """
    Computes the Fourier transform of the signal x.

    Args:
        x: The signal to be transformed.

    Returns:
        The Fourier transform of x.
    """

    N = len(x)
    y = np.fft.fft(x)
    f = np.fft.fftfreq(N)

    return y, f

# Define the path to the CSV file.
csv_file_path = "./WMT.csv"

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

# Take the first day
d1 = d1[:388]
#print(d1.index[387])

# Slice off the Close
X = d1['Close']

# Create a DataPipe to read the CSV file.
#np_data = np.loadtxt(csv_file_path, dtype=np.float32, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6))

# Scale
min_value = np.min(X)
max_value = np.max(X)
Xs = ((X - min_value)/(max_value - min_value) - 0.5)*2.0
Xsf = fft(Xs)/len(Xs)
Xsf = Xsf[range(int(len(Xs)/2))] # exclude sampling frequencey
Xsfa = np.abs(Xsf)
N = len(Xsfa)
n = np.arange(N)
freq = np.fft.fftfreq(n.shape[-1])

print(np.shape(n))
print(np.shape(Xsfa))

plt.figure(figsize = (12, 6))

plt.subplot(121)
plt.stem(n,Xsf.real, 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')

if False:
    plt.subplot(122)
    plt.plot(n, Xs, 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()

plt.show()

exit(0)

if __name__ == "__main__":
    x = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))
    y, f = fourier_transform(x)

    plt.plot(f, np.abs(y))
    plt.show()
