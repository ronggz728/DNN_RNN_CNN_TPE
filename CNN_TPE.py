import numpy as np
import pandas as pd
from pandas import DataFrame
from keras import optimizers
from keras.layers import Dense,Conv1D,MaxPool1D,Flatten,Dropout,Activation
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.compat.v1 import set_random_seed
from hyperopt import fmin,tpe,hp,partial,Trials,STATUS_OK
import loss_history as loss_history

set_random_seed(6)
np.random.seed(6)

train_x = np.load('F://train_x.npy')
test_x = np.load('F://test_x.npy')
train_x = np.expand_dims(train_x,axis=2)
test_x = np.expand_dims(test_x,axis=2)

train_data_y = np.append(np.zeros(336,dtype=int),np.ones(336,dtype=int))
train_y_1D = train_data_y
test_data_y = np.append(np.zeros(144,dtype=int),np.ones(144,dtype=int))
test_y_1D = test_data_y
train_y = np_utils.to_categorical(train_data_y,2)
test_y = np_utils.to_categorical(test_data_y,2)

def percept(args):
    global train_x,train_y,test_y
    ppn = Sequential()
    ppn.add(Conv1D(filters=int(args["filters"]),kernel_size=int(args["kernel_size"]),activation='relu', input_shape=(17, 1)))
    ppn.add(MaxPool1D(pool_size = int(args["pool_size"])))
    ppn.add(Flatten())
    ppn.add(Dense(units=int(args["units"]), activation='relu'))
    ppn.add(Dropout(rate = args['rate']))
    ppn.add(Dense(2, activation='softmax'))

    Adam= optimizers.Adam(learning_rate=0.001)
    Adamax = optimizers.Adamax(learning_rate=0.002)
    SGD = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    RMSprop = optimizers.RMSprop(learning_rate=0.001,rho=0.9)
    Adagrad = optimizers.Adagrad(learning_rate=0.01)
    Adadelta = optimizers.Adadelta(learning_rate=1.0,rho=0.95)
    Nadam = optimizers.Nadam(learning_rate=0.002)

    optimizer = Adam or Adamax or SGD or RMSprop or Adagrad or Adadelta or Nadam

    ppn.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
    history = loss_history.LossHistory()

    ppn.fit(train_x, train_y, validation_data= (test_x, test_y), verbose=2, callbacks=[history], epochs = int(args["epochs"]))
    cost, accuracy = ppn.evaluate(test_x, test_y, verbose=False)
    return -accuracy

space = dict(filters = hp.quniform("filters", 1,50,1), kernel_size =hp.quniform("kernel_size",1,10,2), units = hp.quniform("units", 10,100,1),
             pool_size = hp.quniform("pool_size", 1,10,1), rate=hp.uniform('rate', 0, 0.5), epochs =hp.quniform("epochs", 10,200,1),
             optimizer = {'optimizer' : hp.choice('optimizer',[{'optimizer': 'adam'},{'optimizer': 'adamax'},{'optimizer': 'sgd'},{'optimizer': 'rmsprop'},{'optimizer': 'adagrad'},{'optimizer': 'adadelta'},{'optimizer': 'nadam'}])})

# Trials object to track progress
bayes_trials = Trials()
MAX_EVALS = 500

# Optimize
algo = partial(tpe.suggest,n_startup_jobs=100)
best = fmin(fn = percept, space = space, algo = algo, max_evals = MAX_EVALS, trials = bayes_trials)


print (best)
print (percept(best))
