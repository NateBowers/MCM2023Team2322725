# Written and tested in Python 3.10.4
# See requirements.txt for the versions of packages needed

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import shutil
import argparse
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from initializeData import readData





RANDOM_SEED = 42
METRICS = [
    'MeanSquaredError',
    'MeanAbsolutePercentageError',
]

def plot_vs_epoch(name, output_path, values1, values2=None, compare=False, display=False):
    plt.close()
    path = str(output_path + '/figures')
    save_path = os.path.join(path, name)
    if not os.path.exists(path):
        os.mkdir(path)
    if not compare:
        plt.plot(values1)
        plt.xlabel("Epoch")
        plt.title(name + "vs. Epoch")
    elif compare:
        plt.plot(values1, label='Predicted')
        plt.plot(values2, label='Actual')
        plt.xlabel("Number of guesses")
        plt.ylabel("Probability")
        plt.legend()
    plt.title(name)
    plt.savefig(save_path)
    if display:
        plt.show()

def save_json(data, name, output_path):
    filename = str(output_path + '/' + name + '.json')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4) 

class TrainingData:
    def __init__(self, path_x, path_y, percent_test=0.2):
        data_x = np.genfromtxt(path_x, delimiter=',', dtype='float64')
        data_y = np.genfromtxt(path_y, delimiter=',', dtype='float64')

        if data_x.shape[0] == data_y.shape[0]:
            pass
        elif data_x.shape[0] == data_y.shape[1] or data_x.shape[1] == data_y.shape[0]:
            data_y = np.transpose(data_y)
        else:
            raise Exception("x and y data arrays do not share a common dimension.")

        # mean, std = data_x.mean(), data_x.std()
        # data_x = (data_x-mean)/std

        # self.mean = mean
        # self.std = std

        train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=float(percent_test),random_state=RANDOM_SEED)
        # val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5,random_state=RANDOM_SEED)

        self.train_x = self.convert(train_x)
        self.train_y = self.convert(train_y)
        self.val_x = self.convert(val_x)
        self.val_y = self.convert(val_y)
        # self.test_x = self.convert(test_x)
        # self.test_y = self.convert(test_y)

    def convert(self, arr):
        tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
        return tensor

    def save_stats(self, path):
        stats = {'mean':float(self.mean), 'std':float(self.std)}
        save_json(stats, 'stats', path)

    def IO_shape(self):
        input_shape = self.train_x.shape[1]
        output_shape = self.train_y.shape[1]
        return (input_shape, output_shape)



def main(x_path, y_path, output_path, num_epochs, predictionWord):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    data = TrainingData(x_path, y_path)
    train_x = data.train_x
    train_y = data.train_y
    val_x = data.val_x
    val_y = data.val_y
    # test_x = data.test_x
    # test_y = data.test_y

    # data.save_stats(output_path)
    (input_shape, output_shape) = data.IO_shape()


    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(units=48, activation='relu'))
    model.add(layers.Dense(units=48, activation='relu'))
    model.add(layers.Dense(units=48, activation='relu'))
    model.add(layers.Dense(units=output_shape))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0014),
        loss='mean_squared_error',
        metrics=METRICS,
    )

    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data = (val_x, val_y),
        batch_size=32,
        epochs=num_epochs,
        verbose=0,
    )

    print(model.summary)
    scores = model.evaluate(data.val_x, data.val_y, return_dict=True)



    model.save(os.path.join(output_path, 'model'))
    save_json(history.history, 'histories', output_path)
    save_json(scores, 'scores', output_path)
    for key in history.history:
        plot_vs_epoch(key, output_path, history.history[key])


    wordArr = list(predictionWord)
    word = [0,0,0,0,0]
    for j, letter in enumerate(wordArr):
            # print(j)
            num = ord(letter)-96
            try:
                word[j]=num
            except:
                print(word)

    input_val = tf.reshape(word, shape=[1, data.val_x.shape[1]])
    predicted = model.predict(input_val).flatten()
    predicted = predicted / np.linalg.norm(predicted)
    print(predicted)
    plot_vs_epoch('EERIE', output_path, predicted, display=True)

    
        
    # input_val = tf.reshape(data.test_x[0], shape=[1, data.test_x.shape[1]])
    # predicted = model.predict(input_val).flatten()
    # actual = data.test_y[0].numpy().flatten()
    # plot_vs_epoch('Prediction', output_path, predicted, actual, compare=True, display=True)


if __name__=="__main__":    
    data = readData('Problem_C_Data_Wordle.csv')
    for index, row in enumerate(data): 
        if len(row[2]) != 5:
            data = np.delete(data, index, 0)


    words = data[:,2]
    input = np.empty([data.shape[0], 5])

    for i, word in enumerate(words):
        wordArr = list(word)
        numArr = [0,0,0,0,0]
        for j, letter in enumerate(wordArr):
            # print(j)
            num = ord(letter)-96
            try:
                input[i][j] = num
            except:
                print(word)
                
    input = input / 26

    # print(words)

    output = data[:,5:11]
    print(input.shape)
    print(output.shape)

    # print(input)
    # print(output)


    inputData = 'input.csv'
    outputData = 'output.csv'

    if os.path.exists(inputData):
        os.remove(inputData)
    if os.path.exists(outputData):
        os.remove(outputData)

    np.savetxt(inputData, input, delimiter=',')
    np.savetxt(outputData, output, delimiter=',')

    main(inputData, outputData, 'network', 50, 'eerie')
    

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--x_path",type=str,default='data/8_layer_tio2_val.csv')
    # parser.add_argument("--y_path",type=str,default='data/8_layer_tio2.csv')
    # parser.add_argument("--output_path",type=str,default='results')
    # parser.add_argument("--num_epochs",type=int,default='50')

    # args = parser.parse_args()
    # arg_dict = vars(args)

    # kwargs = {  
    #     'x_path':arg_dict['x_path'],
    #     'y_path':arg_dict['y_path'],
    #     'output_path':arg_dict['output_path'],
    #     'num_epochs':arg_dict['num_epochs'],
    # }

    # main(**kwargs)