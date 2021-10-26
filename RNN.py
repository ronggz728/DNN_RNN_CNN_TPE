#!/usr/bin/python
# # -*- coding=utf-8 -*-
import csv
import random

from pandas import DataFrame
from sklearn import metrics
from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.models import Sequential,load_model
from keras.layers import Dense, SimpleRNN, Activation, BatchNormalization, Dense, Dropout
import loss_history as loss_history
import evaluate_method as evaluate_method

from keras import optimizers
from tensorflow.compat.v1 import set_random_seed
set_random_seed(6)
np.random.seed(6)

test = pd.read_csv('F://test.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC', 'class'])

train = pd.read_csv('F://train.csv', header=None, sep=',',
                 names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC', 'class'])
target = 'class'
data = 'fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan', 'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'

x_columns_train = [x for x in train.columns if x not in [target]]
train_x = train[x_columns_train]
train_y_1D = train['class']

x_columns_test = [x for x in test.columns if x not in [target]]
test_x = test[x_columns_test]
test_y_1D = test['class']

# Full grid data import
dataset_1 = pd.read_csv('F://input_1.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_1 = [x for x in dataset_1.columns]
x_1 = dataset_1[x_columns_1]

dataset_2 = pd.read_csv('F://input_2.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_2 = [x for x in dataset_2.columns]
x_2 = dataset_2[x_columns_2]

dataset_3 = pd.read_csv('F://input_3.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_3 = [x for x in dataset_3.columns]
x_3 = dataset_3[x_columns_3]

dataset_4 = pd.read_csv('F://input_4.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_4 = [x for x in dataset_4.columns]
x_4 = dataset_4[x_columns_4]

dataset_5 = pd.read_csv('F://input_5.csv', header=None, sep=',',
                     names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                        'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])

x_columns_5 = [x for x in dataset_5.columns]
x_5 = dataset_5[x_columns_5]



train_y = np_utils.to_categorical(train_y_1D, 2)
test_y = np_utils.to_categorical(test_y_1D, 2)

train_x = np.expand_dims(train_x,axis=2)
test_x = np.expand_dims(test_x,axis=2)

inputdata_x1 = np.expand_dims(x_1,axis=2)
inputdata_x2 = np.expand_dims(x_2,axis=2)
inputdata_x3 = np.expand_dims(x_3,axis=2)
inputdata_x4 = np.expand_dims(x_4,axis=2)
inputdata_x5 = np.expand_dims(x_5,axis=2)

model = Sequential()
model.add(SimpleRNN(50, batch_input_shape=(None, 17, 1), unroll=True))
model.add(Dropout(0))
model.add(Dense(2))
model.add(Activation('softmax'))
optimizer = optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# Fit the model
print(model.summary())
history= loss_history.LossHistory()
model.fit(train_x,train_y,validation_data= (test_x,test_y),verbose=2,callbacks=[history],batch_size=64,epochs=50)

y_prob_test = model.predict(test_x)    
y_probability_first = [prob[1] for prob in y_prob_test]

predicted = model.predict(test_x)
exp = DataFrame(predicted)
exp.to_csv("F://RNN.csv")
print(predicted)

print(metrics.classification_report(test_y_1D, y_probability_first))
print(metrics.confusion_matrix(test_y_1D, y_probability_first))

model.save('my_model_RNN.h5')
history.loss_plot('epoch')

probability_1 = model.predict(inputdata_x1)
exp_1 = DataFrame(probability_1)
exp_1.to_csv("F://RNN//RNN_ALL_1.csv")
probability_2 = model.predict(inputdata_x2)
exp_2 = DataFrame(probability_2)
exp_2.to_csv("F://RNN//RNN_ALL_2.csv")
probability_3 = model.predict(inputdata_x3)
exp_3 = DataFrame(probability_3)
exp_3.to_csv("F://RNN//RNN_ALL_3.csv")
probability_4 = model.predict(inputdata_x4)
exp_4 = DataFrame(probability_4)
exp_4.to_csv("F://RNN//RNN_ALL_4.csv")
probability_5 = model.predict(inputdata_x5)
exp_5 = DataFrame(probability_5)
exp_5.to_csv("F://RNN//RNN_ALL_5.csv")





