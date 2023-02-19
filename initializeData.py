# Scripts for reading data and making basic calculations on it.

import numpy as np
import pandas as pd

newColNames = ['date', 'contestNum', 'word', 'numResults', 'numHard', 'oneTry', 'twoTry', 'threeTry', 'fourTry', 'fiveTry', 'sixTry', 'sevenTry']

def readData(filename, colNames=newColNames, pdArray=False):
    df = pd.read_csv(str(filename))
    # rename columns
    df.columns=colNames
    # add means
    df['means'] = guessStats(df.values)
    # can output as either numpy array or pandas dataframe, default is nparray
    if pdArray:
        return df
    else:
        return df.values

def guessStats(data):
    tries = data[:,5:12].T
    tries = tries / 100
    mean = (tries[0]) + (2 * tries[1]) + (3 * tries[2]) + (4* tries[3]) + (5 * tries[4]) + (6 * tries[5]) + (7 * tries[6])
    return mean.astype('float64')




if __name__ == '__main__':
    data = readData('Problem_C_Data_Wordle.csv')
    print(data)

    # words = data.T[2]

    # print(words)
    # print(type(words))

    # for word in words:
    #      if len(word) != 5:
    #           print("'"+ word+ "'" + " has length of: " + str(len(word)))