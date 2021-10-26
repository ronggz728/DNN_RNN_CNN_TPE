import numpy as np
from pandas import DataFrame

test_blending_x_new_train = np.load('test_blending_y_new_train.npy')
exp = DataFrame(test_blending_x_new_train)
exp.to_csv("test_blending_x_new_train.csv")
print(test_blending_x_new_train)