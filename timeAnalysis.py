# Discrete Fourier Transform to see if there are any periodic trends within the data.

import numpy as np
from initializeData import readData

data = readData('Problem_C_Data_Wordle.csv')

# get rid of unnecessary data and make the start date 0

data = np.delete(data, [0,2,3,4], axis=1)
print(data)
print(data.shape)