import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#def func(x, amplitude, frequency, slope, a, b):
#  return slope*x + (a * np.exp(-b * x))*amplitude * np.sin(2 * np.pi * frequency * x)

def func(x, slope): # amplitude, frequency, slope, a, b):
# return slope*x + (a * np.exp(-b * x))*amplitude * np.sin(2 * np.pi * frequency * x)
  return slope*x

def scale_tensor(tensor):
    # Compute the maximum and minimum values of the tensor
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    
    # Scale the tensor from -1 to +1
    scaled_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
    
    return scaled_tensor

Y = [ 82.7300,82.8000,82.7200,82.7600,83.1800,83.1150,83.0700,83.0600,83.1100,83.0500,
83.1300,83.1600,83.0400,82.9900,82.9100,82.9700,83.0300,83.0843,83.2000,83.1850,
83.0400,83.1400,83.1962,83.1000,83.1450,83.1100,83.2450,83.3200,83.3400,83.3600,
83.3950,83.4650,83.4400,83.4400,83.4900,83.4350,83.4300,83.4300,83.4800,83.4750,
83.4550,83.5000,83.5400,83.5000,83.5600,83.6235,83.5800,83.5600,83.5900,83.6300,
83.6100,83.6400,83.6200,83.5400,83.5500,83.6000,83.5300,83.4900,83.5250,83.5650,
83.4900,83.4700,83.4717,83.4300,83.4600,83.4700,83.4700,83.4200,83.4450,83.4950,
83.4658,83.4300,83.3900,83.3800,83.3800,83.2500,83.2800,83.3700,83.3400,83.3195,
83.2500,83.2000,83.1500,83.1700,83.1648,83.1000,83.1400,83.0800,83.1150,83.1800,
83.1000,83.0500,83.0098,83.0000,83.0700,83.0700,83.0300,83.0100,83.0500,83.0500,
83.1000,83.1200,83.1300,83.1500,83.1700,83.1800,83.1800,83.1850,83.1400,83.0950,
83.1967,83.2100,83.2300,83.2300,83.2600,83.3100,83.3300,83.2300,83.1800,83.1900,
83.2300,83.1700,83.1826,83.1274,83.0880,83.1050,83.1000,83.1000,83.1200,83.1400,
83.1750,83.1400,83.1200,83.1300,83.1100,83.1200,83.1200,83.1500,83.1700,83.1699,
83.2400,83.2900,83.3000,83.3531,83.3500,83.3900,83.4000,83.4200,83.3750,83.3700,
83.4500,83.4200,83.3200,83.4300,83.4600,83.4700,83.4700,83.4350,83.4200,83.4300,
83.4500,83.4725,83.4900,83.4685,83.4800,83.4900,83.5100,83.5500,83.5650,83.5500,
83.5750,83.5800,83.5500,83.5500,83.4800,83.4300,83.3900,83.4000,83.3461,83.3750,
83.4150,83.4300,83.4400,83.4600,83.4300,83.4100,83.3800,83.4300,83.3650,83.3500,
83.3500,83.2750,83.3050,83.2870,83.2600,83.3200,83.2930,83.2800,83.2500,83.2500,
83.2350,83.2500,83.2100,83.2450,83.2300,83.2250,83.2100,83.2100,83.2300,83.2250,
83.1900,83.2000,83.2350,83.2650,83.2450,83.2300,83.2386,83.2401,83.2450,83.2300,
83.2800,83.2600,83.2600,83.2900,83.2804,83.3000,83.3500,83.3500,83.3550,83.3700,
83.3850,83.4050,83.4350,83.4050,83.3950,83.3750,83.3650,83.3800,83.3600,83.3550,
83.3750,83.3850,83.3750,83.3850,83.3550,83.3400,83.2900,83.2850,83.3550,83.3700,
83.3800,83.3950,83.3650,83.3450,83.3500,83.3650,83.3400,83.3400,83.3250,83.3200,
83.3300,83.3450,83.3450,83.3800,83.3850,83.4500,83.4400,83.4600,83.4200,83.4150,
83.4250,83.4250,83.4600,83.4350,83.4400,83.4600,83.4800,83.4650,83.4300,83.3850,
83.4100,83.4083,83.3950,83.3950,83.4160,83.4250,83.4500,83.4600,83.4550,83.4650,
83.4500,83.4500,83.4450,83.4450,83.4200,83.4250,83.4250,83.4183,83.4450,83.4400,
83.4400,83.4500,83.4500,83.4250,83.3800,83.3700,83.3300,83.3300,83.3330,83.3100,
83.3550,83.3600,83.3800,83.3598,83.3300,83.3500,83.3300,83.3150,83.2800,83.2800,
83.3100,83.3400,83.3400,83.3152,83.3000,83.3300,83.3200,83.3200,83.3200,83.3550,
83.3550,83.3550,83.3000,83.3100,83.3150,83.3550,83.3350,83.3350,83.3350,83.3400,
83.3450,83.3600,83.3750,83.4100,83.3700,83.4400,83.4250,83.4450,83.4400,83.4300,
83.4300,83.4450,83.4450,83.4250,83.4200,83.3900,83.3850,83.3900,83.3800,83.4050,
83.3800,83.4000,83.3950,83.3850,83.3900,83.3950,83.3850,83.4200,83.4200,83.4100,
83.4050,83.3950,83.4000,83.3550,83.3550,83.3700,83.3450,83.3350,83.3100,83.2950,
83.3150,83.3050,83.3050,83.3200,83.3400,83.3600,83.3750,83.4200,83.4150,83.5600 ]

y = scale_tensor(Y)
x = np.linspace(0, 6.28, 390) # generate 390 samples evenly spaced between 0 and 1

#popt, pcov = curve_fit(func, x, y, bounds=([0.8,3.0,-2.0, 0.0, 0.0], [1., 8., 2.0,3.0,1.0]))
popt, pcov = curve_fit(func, x, y, bounds=(-2.0, [2.0]))
print(popt)

if True:
  plt.plot(x, y, 'b-', label='data')

  plt.plot(x, func(x, *popt), 'r-',
           #         label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' % tuple(popt))
           label='fit: a=%5.3f' % tuple(popt))         

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

slope = popt[0]
print(f"slope = {slope}")

#now calculate the y value given x and the slope, and subtract that from y
for idx in range(len(x)):
  y[idx] -= slope*x[idx]

if True:
  plt.plot(x, y, 'g-',
           label='slope adjusted')

# now average
new_y = []
for idx in range(len(x)):
  if idx == 0 or idx == 1:
    new_y.append(y[idx])
    continue
  new_y.append((y[idx-2] + y[idx-1] + y[idx])/3)

if True:
  plt.plot(x, new_y, 'g-',
           label='Adjusted')

# match up sin wave
def func1(x, amplitude, frequency):
  return amplitude * np.sin(frequency * x)

#print(new_y)

popt, pcov = curve_fit(func1, x, new_y, bounds=([0.8, 3.0], [1.2, 5.0]))

print(popt)
#print(x)

plt.plot(x, func1(x, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

# Frequency domain representation

fourierTransform = np.fft.fft(new_y)/len(new_y)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(new_y)/2))] # Exclude sampling frequency

#fourierTransform = abs(fourierTransform)
#m = max(fourierTransform)
#print(m)
# scale
#fourierTransform *= 1/m
#print(fourierTransform)
print(fourierTransform)
exit(0)

#fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency

#tpCount     = len(amplitude)

#values      = np.arange(int(tpCount/2))

#timePeriod  = tpCount/samplingFrequency

#frequencies = values/timePeriod

# Frequency domain representation

#plt.set_title('Fourier transform depicting the frequency components')
#plt.plot(x, fourierTransfor)
#plt.set_xlabel('Frequency')
#plt.set_ylabel('Amplitude')
 
def func2(x):
  fft = (0.27390225, 0.292318, 0.49359872, 1., 0.94604817, 0.32651918, 0.51503608, 0.23864754, 0.06561233, 0.14582141, 0.218635, 0.08602998)
  res = 0.0
  for i in (1,2,3,4,5,6,7,8,9,10,11):
    res += fft[i]*np.sin(x*i)
  return res

yff = func2(x)
yff = yff/max(yff)

print(yff)
plt.plot(x, yff)

plt.show()
