#import torch
import numpy as np
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

# Create a DataPipe to read the CSV file.
np_data = np.loadtxt(csv_file_path, dtype=np.float32, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6))

# Slice off the close
x = np_data_close = np_data[:,3]
xr = x[10:-10]
X = fft(x)
Xr = np.log10(np.abs(X[10:-10]))
N = len(Xr)
n = np.arange(N)
T = N/1
freq = n/T

#print(f"x = {x}")
#print(f"Xr = {Xr}")
#print(f"N = {N}")

#plt.plot(x)
#plt.show()
#
#exit(0)

#plt.plot(n,Xr)
#plt.show()
#exit(0)

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(n,Xr, 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
#plt.xlim(0, 10)

plt.subplot(122)
plt.plot(n, xr, 'r')
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
