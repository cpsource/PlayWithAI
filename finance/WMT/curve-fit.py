import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x, amplitude, frequency):
  return -1 * amplitude * np.sin(2 * np.pi * frequency * x)

if __name__ == "__main__":

  amplitude = 1
  frequency = 1
  
  xdata = np.linspace(0, 1, 100) # generate 50 samples evenly spaced between 0 and 2*pi
  y = func(xdata, amplitude, frequency)
  rng = np.random.default_rng()
  y_noise = 0.2 * rng.normal(size=xdata.size)
  ydata = y_noise + y

  plt.figure(figsize = (12, 6))  
  plt.plot(xdata, ydata, 'b-', label='data')

  popt, pcov = curve_fit(func, xdata, ydata)
  print(popt)
  plt.plot(xdata, func(xdata, *popt), 'r-',
           label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  plt.show()

  if False:
    plt.subplot(121)
    plt.stem(n,X[2], 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')

  if False:
    #plt.subplot(122)
    plt.plot(x, y, 'r')
    plt.xlabel('frequence')
    plt.ylabel('amplitude')
    plt.tight_layout()
      
    #plt.show(block=False)
    plt.show()
    #plt.pause(5)
    #plt.close()
  
    #print(params)

