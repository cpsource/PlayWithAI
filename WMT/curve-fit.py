import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def curve_fit_sine(x, y, amplitude, frequency):
  """
  Curve-fits data to a sine wave with given amplitude and frequency.

  Args:
    x: The x-coordinates of the data.
    y: The y-coordinates of the data.
    amplitude: The amplitude of the sine wave.
    frequency: The frequency of the sine wave.

  Returns:
    The fitted parameters of the sine wave.
  """

  def sine_func(x, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * x)

  params, other = curve_fit(sine_func, x, y)
  """
  Curve-fits data

  Args:
    sine_func: f(x...)
    x: The independant x-coordinates of the data.
    y: The dependant y-coordinates of the data.

  Returns:

  """

  print(f"params = {params}, other = {other}")
  
  return params

if __name__ == "__main__":

  amplitude = 5
  frequency = 2
  offset = 10
  
  x = np.linspace(0, 10, 100) # generate 100 samples evenly spaced between 0 and 10

  print(f"x = {x}")

  y = amplitude * np.sin(2 * np.pi * frequency * x) + offset

  print(f"y = {y}")

  params = curve_fit_sine(x, y, amplitude, frequency)

  plt.figure(figsize = (12, 6))

  if False:
    plt.subplot(121)
    plt.stem(n,X[2], 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')

  if True:
    #plt.subplot(122)
    plt.plot(x, y, 'r')
    plt.xlabel('frequence')
    plt.ylabel('amplitude')
    plt.tight_layout()
      
    #plt.show(block=False)
    plt.show()
    #plt.pause(5)
    #plt.close()
  
  print(params)

