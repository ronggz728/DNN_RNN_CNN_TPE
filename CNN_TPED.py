#!/usr/bin/python
# -*- coding=utf-8 -*-
from pandas import DataFrame
from sklearn import metrics
from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPool1D,Flatten,Dropout
from keras import optimizers
import loss_history as loss_history
import evaluate_method as evaluate_method
from tensorflow.compat.v1 import set_random_seed
set_random_seed(6)
np.random.seed(6)
#read train data

train_x = np.load('F://第三篇//样本处理//CNN//2倍//train_x.npy')
test_x = np.load('F://第三篇//样本处理//CNN//2倍//test_x.npy')
train_x = np.expand_dims(train_x,axis=2)
test_x = np.expand_dims(test_x,axis=2)

train_data_y = np.append(np.ones(336,dtype=int),np.zeros(336,dtype=int))
train_y_1D = train_data_y
test_data_y = np.append(np.ones(144,dtype=int),np.zeros(144,dtype=int))
test_y_1D = test_data_y
train_y = np_utils.to_categorical(train_data_y,2)
test_y = np_utils.to_categorical(test_data_y,2)



model = Sequential()
#添加输入层
model.add(Conv1D(20,6,activation='relu',input_shape=(17,1)))
model.add(MaxPool1D(2))
model.add(Flatten())
model.add(Dense(69,activation='relu'))
model.add(Dropout(0.17740603719035558))
model.add(Dense(2,activation='softmax'))
optimizer = optimizers.RMSprop(learning_rate=0.001,rho=0.9)
model.compile(loss='categorical_crossentropy',
             optimizer=optimizer,metrics=['accuracy'])
print(model.summary())


history = loss_history.LossHistory()
model.fit(train_x,train_y,validation_data= (test_x,test_y),verbose=2,callbacks=[history],epochs=177)

with open('train_acc_cnn.txt', 'w') as fp:
    for loss in range(len(history.val_acc['epoch'])):
        fp.write(str(loss+1)  + ',' + str(history.accuracy['epoch'][loss]) +  '\n')

with open('test_acc_cnn.txt', 'w') as fp:
    for loss in range(len(history.val_acc['epoch'])):
        fp.write(str(loss+1)  + ',' + str(history.val_acc['epoch'][loss]) +  '\n')

with open('train_err_cnn.txt', 'w') as fp:
    for loss in range(len(history.val_loss['epoch'])):
        fp.write(str(loss+1)  + ',' + str(history.losses['epoch'][loss]) +  '\n')

with open('test_err_cnn.txt', 'w') as fp:
   for loss in range(len(history.val_loss['epoch'])):
       fp.write(str(loss+1)  + ',' + str(history.val_loss['epoch'][loss]) +  '\n')


y_prob_test = model.predict(test_x)     #output predict probability
y_probability_first = [prob[1] for prob in y_prob_test]

exp = DataFrame(y_prob_test)
exp.to_csv("F://CNN_TPE.csv")
print(y_prob_test)

print(metrics.classification_report(test_y_1D, y_probability_first))
print(metrics.confusion_matrix(test_y_1D, y_probability_first))

model.save('my_model_CNN1.h5')
history.loss_plot('epoch')

# Full grid data import
input_1 = np.load('F://input_1.npy')
input_2 = np.load('F://input_2.npy')
input_3 = np.load('F://input_3.npy')
input_4 = np.load('F://input_4.npy')
input_5 = np.load('F://input_5.npy')
input_1 = np.expand_dims(input_1,axis=2)
input_2 = np.expand_dims(input_2,axis=2)
input_3 = np.expand_dims(input_3,axis=2)
input_4 = np.expand_dims(input_4,axis=2)
input_5 = np.expand_dims(input_5,axis=2)

output_1 = model.predict(input_1)
output_2 = model.predict(input_2)
output_3 = model.predict(input_3)
output_4 = model.predict(input_4)
output_5 = model.predict(input_5)

exp_1 = DataFrame(output_1)
exp_1.to_csv("F://CNN_TPE_1.csv")
exp_2 = DataFrame(output_2)
exp_2.to_csv("F://CNN_TPE_2.csv")
exp_3 = DataFrame(output_3)
exp_3.to_csv("F://CNN_TPE_3.csv")
exp_4 = DataFrame(output_4)
exp_4.to_csv("F://CNN_TPE_4.csv")
exp_5 = DataFrame(output_5)
exp_5.to_csv("F://CNN_TPE_5.csv")