# Discrete Fourier Transform to see if there are any periodic trends within the data.

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt
from initializeData import readData

data = readData('Problem_C_Data_Wordle.csv')

# get rid of unnecessary data and make the start date 0

data = np.delete(data.T, [0,1,2,3,4], axis=0)
# print(data)
print(data.shape)

testData = data[7]

# plt.plot(testData)
# plt.show()


    

def fourier(data):
    dataNorm = data - np.mean(data)
    n = len(dataNorm)
    yf = rfft(dataNorm)
    xf = rfftfreq(n, 1/n)
    yfAbs = np.abs(yf)
    yfNorm = yfAbs / np.linalg.norm(yfAbs)
    return xf, yfNorm



noise = np.random.normal(0,.5,100)
foo = np.linspace(1,50, num=100)
bar = np.sin(foo)
bar2 = np.sin(foo*4)
bar3 = bar+bar2
plt.plot(bar3)
plt.show()

baz = bar3 + noise
plt.plot(baz)
plt.show()

example = fourier(baz)
plt.plot(example[0], example[1])
plt.show()

auto = pd.plotting.autocorrelation_plot(baz)
auto.plot()
plt.show()

# for row in data:

row = data[7]
autoCorPlt = pd.plotting.autocorrelation_plot(row)
autoCorPlt.plot()
plt.show()
rowTransform = fourier(row)
plt.plot(rowTransform[0], rowTransform[1])
plt.show()

