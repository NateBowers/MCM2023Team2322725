# Discrete Fourier Transform to see if there are any periodic trends within the data.

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt
from initializeData import readData

data = readData('Problem_C_Data_Wordle.csv')

# get rid of unnecessary data and make the start date 0

data = np.delete(data.T, [0,1,2,3,4], axis=0)
    
def fourier(data):
    dataNorm = data - np.mean(data)
    n = len(dataNorm)
    yf = rfft(dataNorm)
    xf = rfftfreq(n, 1/n)
    yfAbs = np.abs(yf)
    yfNorm = yfAbs / np.linalg.norm(yfAbs)
    return xf, yfNorm


row = data[7]
autoCorPlt = pd.plotting.autocorrelation_plot(row)
autoCorPlt.plot()
plt.savefig('Autocurrelation', dpi=300)
plt.show()

rowTransform = fourier(row)
plt.plot(rowTransform[0], rowTransform[1])
plt.savefig('Fourier',  dpi=300)
plt.show()

