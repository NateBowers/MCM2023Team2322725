# Discrete Fourier Transform to see if there are any periodic trends within the data.

import numpy as np
from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt
from initializeData import readData

data = readData('Problem_C_Data_Wordle.csv')

# get rid of unnecessary data and make the start date 0

data = np.delete(data.T, [0,1,2,3,4], axis=0)
# print(data)
print(data.shape)

testData = data[7]

plt.plot(testData)
plt.show()


    

def fourier(data):
    dataNorm = data - np.mean(data)
    n = len(dataNorm)
    yf = rfft(dataNorm)
    xf = rfftfreq(n, 1/n)
    yfAbs = np.abs(yf)
    yfNorm = yfAbs / np.linalg.norm(yfAbs)
    return xf, yfNorm



for row in data:
    rowTransform = fourier(row)
    plt.plot(rowTransform[0], rowTransform[1])
    plt.show()
    print(row)
