import loss_history as loss_history
import numpy as np
import pandas as pd
from pandas import DataFrame
import keras
from keras.layers import Dropout
from tensorflow.compat.v1 import set_random_seed
from hyperopt import fmin, tpe, hp, partial, Trials
from sklearn.preprocessing import StandardScaler  # 标准化工具



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
train_y = train['class']

x_columns_test = [x for x in test.columns if x not in [target]]
test_x = test[x_columns_test]
test_y = test['class']

# 转化为list
x1_array = np.array(train_x)
y1_array = np.array(train_y)
x2_array = np.array(test_x)
y2_array = np.array(test_y)

X1_list = x1_array.tolist()
Y1_list = y1_array.tolist()
X2_list = x2_array.tolist()
Y2_list = y2_array.tolist()

# 标准化
scaler = StandardScaler()
scaler.fit(X1_list)
scaler.fit(X2_list)

X_train = scaler.transform(X1_list)
X_test = scaler.transform(X2_list)

Y_train = np.array(Y1_list)
Y_test = np.array(Y2_list)

# 定义模型
init = keras.initializers.he_normal(seed=1)


def percept(args):
    global X_train, Y_train, X_test, Y_test

    Adam = keras.optimizers.Adam(learning_rate=0.001)
    Adamax = keras.optimizers.Adamax(learning_rate=0.002)
    SGD = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    RMSprop = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    Adagrad = keras.optimizers.Adagrad(learning_rate=0.01)
    Adadelta = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    Nadam = keras.optimizers.Nadam(learning_rate=0.002)
    optimizer = Adam or Adamax or SGD or RMSprop or Adagrad or Adadelta or Nadam

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=int(args["units"]), input_dim=17, activation='relu'))
    model.add(keras.layers.Dense(units=int(args["units"]), activation='relu'))
    model.add(Dropout(rate = args['rate']))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 训练模型
    history = loss_history.LossHistory()
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, callbacks=[history], batch_size = int(args["batch_size"]), epochs=int(args["epochs"]))
    cost, accuracy = model.evaluate(X_test, Y_test, verbose=False)
    return -accuracy


space = dict(units=hp.quniform("units", 10, 100, 1), rate=hp.uniform('rate', 0, 0.5),batch_size = hp.quniform("batch_size", 10,100,1), epochs=hp.quniform("epochs", 10, 100, 1),
             optimizer={'optimizer': hp.choice('optimizer', [{'optimizer': 'adam'}, {'optimizer': 'adamax'},
                                                             {'optimizer': 'sgd'}, {'optimizer': 'rmsprop'},
                                                             {'optimizer': 'adagrad'}, {'optimizer': 'adadelta'},
                                                             {'optimizer': 'nadam'}])})

# Trials object to track progress
bayes_trials = Trials()
MAX_EVALS = 500

# Optimize
algo = partial(tpe.suggest, n_startup_jobs=100)
best = fmin(fn=percept, space=space, algo=algo, max_evals=MAX_EVALS, trials=bayes_trials)

print(best)
print(percept(best))
