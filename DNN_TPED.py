# -*- coding: utf-8 -*-

import numpy as np
import keras
from sklearn import metrics
from keras.callbacks import EarlyStopping  # 防止过拟合
from keras.layers import Dropout
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler  # 标准化工具
from sklearn.metrics import classification_report
import loss_history as loss_history
import evaluate_method as evaluate_method

def main():
    #dataset 导入
    print("Loading data into memory")


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

    #转化为list
    x1_array = np.array(train_x)
    y1_array = np.array(train_y)
    x2_array = np.array(test_x)
    y2_array = np.array(test_y)

    X1_list = x1_array.tolist()
    Y1_list= y1_array.tolist()
    X2_list = x2_array.tolist()
    Y2_list = y2_array.tolist()

    #标准化
    scaler = StandardScaler()
    scaler.fit(X1_list)
    scaler.fit(X2_list)

    X_train = scaler.transform(X1_list)
    X_test = scaler.transform(X2_list)

    Y_train = np.array(Y1_list)
    Y_test = np.array(Y2_list)

    # 定义模型
    init = keras.initializers.he_normal(seed=1)

    adam = keras.optimizers.Adam()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=96, input_dim=17, kernel_initializer=init, activation='relu'))
    model.add(keras.layers.Dense(units=96, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.3945938575437625))
    model.add(keras.layers.Dense(units=1, kernel_initializer=init, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    # 训练模型
    print("Starting training ")
    history = loss_history.LossHistory()
    model.fit(X_train, Y_train, shuffle=True, validation_data=(X_test, Y_test), verbose=1, callbacks=[history], batch_size=38, epochs=42)

    print("Training finished \n")

    #测试
    pre_test = model.predict(X_test)
    exp = DataFrame(pre_test)
    exp.to_csv("F://DNN_TPE.csv")
    print(pre_test)
    
    for i in range(len(pre_test)):
        if pre_test[i] <= 0.5:
            pre_test[i] = 0
        else:
            pre_test[i] = 1

    print(classification_report(Y_test, pre_test))

    print(metrics.classification_report(Y_test, pre_test))
    print(metrics.confusion_matrix(Y_test, pre_test))
    
    model.save('my_model_DNN1.h5')
    history.loss_plot('epoch')

    with open('train_acc_rnn.txt', 'w') as fp:
        for loss in range(len(history.val_acc['epoch'])):
            fp.write(str(loss + 1) + ',' + str(history.accuracy['epoch'][loss]) + '\n')

    with open('test_acc_rnn.txt', 'w') as fp:
        for loss in range(len(history.val_acc['epoch'])):
            fp.write(str(loss + 1) + ',' + str(history.val_acc['epoch'][loss]) + '\n')

    with open('train_err_rnn.txt', 'w') as fp:
        for loss in range(len(history.val_loss['epoch'])):
            fp.write(str(loss + 1) + ',' + str(history.losses['epoch'][loss]) + '\n')

    with open('test_err_rnn.txt', 'w') as fp:
        for loss in range(len(history.val_loss['epoch'])):
            fp.write(str(loss + 1) + ',' + str(history.val_loss['epoch'][loss]) + '\n')

    # Full grid data import
    dataset_1 = pd.read_csv('F://input_1.csv', header=None, sep=',',
                        names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                             'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])



    dataset_2 = pd.read_csv('F://input_2.csv', header=None, sep=',',
                       names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                             'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])


    dataset_3 = pd.read_csv('F://input_3.csv', header=None, sep=',',
                         names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                             'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])


    dataset_4 = pd.read_csv('F://input_4.csv', header=None, sep=',',
                         names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                                'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])


    dataset_5 = pd.read_csv('F://input_5.csv', header=None, sep=',',
                          names=['fault', 'road', 'river', 'lithology', 'elevation', 'slope', 'NDVI', 'profile', 'plan',
                             'aspect', 'geological', 'rain', 'SPI', 'TWI', 'TRI', 'STI', 'LUCC'])


    data_array_1 = np.array(dataset_1)
    data_list_1 = data_array_1.tolist()
    scaler.fit(data_list_1)
    data_in_1 = scaler.transform(data_list_1)
    pre_test_data_1 = model.predict(data_in_1)

    data_array_2 = np.array(dataset_2)
    data_list_2 = data_array_2.tolist()
    scaler.fit(data_list_2)
    data_in_2 = scaler.transform(data_list_2)
    pre_test_data_2 = model.predict(data_in_2)

    data_array_3 = np.array(dataset_3)
    data_list_3 = data_array_3.tolist()
    scaler.fit(data_list_3)
    data_in_3 = scaler.transform(data_list_3)
    pre_test_data_3 = model.predict(data_in_3)

    data_array_4 = np.array(dataset_4)
    data_list_4 = data_array_4.tolist()
    scaler.fit(data_list_4)
    data_in_4 = scaler.transform(data_list_4)
    pre_test_data_4 = model.predict(data_in_4)

    data_array_5 = np.array(dataset_5)
    data_list_5 = data_array_5.tolist()
    scaler.fit(data_list_5)
    data_in_5 = scaler.transform(data_list_5)
    pre_test_data_5 = model.predict(data_in_5)

    exp_1 = DataFrame(pre_test_data_1)
    exp_1.to_csv("F://第三篇//DNN//ALL_TPE_1.csv")

    exp_2 = DataFrame(pre_test_data_2)
    exp_2.to_csv("F://第三篇//DNN//ALL_TPE_2.csv")

    exp_3 = DataFrame(pre_test_data_3)
    exp_3.to_csv("F://第三篇//DNN//ALL_TPE_3.csv")

    exp_4 = DataFrame(pre_test_data_4)
    exp_4.to_csv("F://第三篇//DNN//ALL_TPE_4.csv")

    exp_5 = DataFrame(pre_test_data_5)
    exp_5.to_csv("F://第三篇//DNN//ALL_TPE_5.csv")


main()