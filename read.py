# Scripts for reading data and making basic calculations on it.

import numpy as np
import pandas as pd


newColNames = ['date', 'contestNum', 'word', 'numResults', 'numHard', 'oneTry', 'twoTry', 'threeTry', 'fourTry', 'fiveTry', 'sixTry', 'sevenTry']


def readData(filename, colNames=newColNames, pdArray=False):
    df = pd.read_csv(str(filename))
    df.columns=newColNames
    if pdArray:
        return df
    else:
        return df.values

def stats(data):
    triesArr = data[:,'oneTry':'sevenTry']
    return triesArr



data = readData('Problem_C_Data_Wordle.csv', pdArray=True)

print(data)

