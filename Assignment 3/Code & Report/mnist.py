#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:35:15 2020

@author: kishor
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
from mnist import MNIST
import numpy as np
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers  import Dense, Conv2D, Flatten
from keras.utils import to_categorical
if __name__ == "__main__":
    file_path=sys.argv[1]
    mndata = MNIST(file_path)
    X_train,Y_train = mndata.load_training()
    X_test,Y_test= mndata.load_testing()
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    X_test=np.array(X_test)
    Y_train_en = to_categorical(Y_train, 10)
    X_train_ = X_train.reshape(60000,28,28,1)
    X_test_ = X_test.reshape(10000,28,28,1)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_,Y_train_en,epochs=100, batch_size=100,verbose=0)
    pre= model.predict(X_test_)
    for i in pre:
        print(np.argmax(i))