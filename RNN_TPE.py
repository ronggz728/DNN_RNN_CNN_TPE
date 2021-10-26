import loss_history as loss_history
import numpy as np
import pandas as pd
from pandas import DataFrame
from keras import optimizers
from keras.layers import SimpleRNN, Activation, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.compat.v1 import set_random_seed
from hyperopt import fmin,tpe,hp,partial,Trials,STATUS_OK

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

train_y = np_utils.to_categorical(train_y_1D, 2)
test_y = np_utils.to_categorical(test_y_1D, 2)

train_x = np.expand_dims(train_x,axis=2)
test_x = np.expand_dims(test_x,axis=2)

def percept(args):
    global train_x,train_y,test_y
    ppn = Sequential()
    ppn.add(SimpleRNN(units = int(args["units"]),batch_input_shape=(None, 17, 1),unroll=True))
    ppn.add(Dropout(rate = args['rate']))
    ppn.add(Dense(2))
    ppn.add(Activation(activation = 'softmax'))

    Adam = optimizers.Adam(learning_rate=0.001)
    Adamax = optimizers.Adamax(learning_rate=0.002)
    SGD = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    RMSprop = optimizers.RMSprop(learning_rate=0.001,rho=0.9)
    Adagrad = optimizers.Adagrad(learning_rate=0.01)
    Adadelta = optimizers.Adadelta(learning_rate=1.0,rho=0.95)
    Nadam = optimizers.Nadam(learning_rate=0.002)
    optimizer = Adam or Adamax or SGD or RMSprop or Adagrad or Adadelta or Nadam
    ppn.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])

    history = loss_history.LossHistory()
    ppn.fit(train_x,train_y,validation_data= (test_x,test_y),verbose=2,callbacks=[history],batch_size = int(args["batch_size"]),epochs = int(args["epochs"]) )
    cost, accuracy = ppn.evaluate(test_x, test_y, verbose=False)
    return -accuracy

space = dict(units = hp.quniform("units", 10,100,1), rate=hp.uniform('rate', 0, 0.5),batch_size = hp.quniform("batch_size", 10,100,1),
             epochs =hp.quniform("epochs", 10,100,1), optimizer =
             {'optimizer' : hp.choice('optimizer',[{'optimizer': 'adam'},{'optimizer': 'adamax'},{'optimizer': 'sgd'},{'optimizer': 'rmsprop'},{'optimizer': 'adagrad'},{'optimizer': 'adadelta'},{'optimizer': 'nadam'}])})

# Trials object to track progress
bayes_trials = Trials()
MAX_EVALS = 500

# Optimize
algo = partial(tpe.suggest,n_startup_jobs=100)
best = fmin(fn = percept, space = space, algo = algo, max_evals = MAX_EVALS, trials = bayes_trials)


print (best)
print (percept(best))

