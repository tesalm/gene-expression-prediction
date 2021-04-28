# -*- coding: utf-8 -*-
"""
DeepChrome: deep-learning for predicting gene expression from histone modifications
https://arxiv.org/pdf/1607.02078.pdf
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
#from keras import backend as k
#k.set_image_dim_ordering('th')

from parse_data import preprocess_data_for_cnn


if __name__ == '__main__':
    x_train, y_train, x_test = preprocess_data_for_cnn()

    # Data normalization (between [0, 1])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    normalized_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    normalized_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    num_classes = 2  # gene has high or low expression level.
    y_train = np_utils.to_categorical(y_train, num_classes)

    # a typical modern convolution network (conv+relu+pool)
    model = Sequential()

    # 1D convolution layer (e.g. temporal convolution).
    # kernel_size: length of the 1D convolution window
    # strides: stride length of the convolution
    model.add(Convolution1D(filters=50, kernel_size=4, strides=1,
                            padding = 'same', 
                            activation = 'relu', 
                            input_shape = (5, 100)))

    # pool_size: size of the max pooling window
    model.add(MaxPooling1D(pool_size=3, strides=1))

    # Dropout randomly sets input units to 0 with a frequency of 'rate' at 
    # each step during training time, which helps prevent overfitting
    model.add(Dropout(rate=0.5))

    # The Flatten() rearranges the output matrix of the MaxPooling layer into 
    # a one-dimensional vector that can be fed to the perseptron layer as input. 
    model.add(Flatten())

    # Perseptron layers (regular deeply connected neural network)
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax'))

    # Compile model using 'accuracy' to measure model performance
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    # Train the model, takes a few minutes
    model.fit(normalized_train, y_train, epochs=50, batch_size=16)

    loss, accuracy = model.evaluate(normalized_train, y_train)

    # Remove the first column (probability of the gene NOT being active)
    pred_prob = model.predict(x_test)[:,1]

    print('Loss: %.3f,  Accuracy: %.2f' % (loss, accuracy*100))
    # Loss: 0.149,  Accuracy: 94.72


    # create submission csv file    
    with open('subm.csv','wb') as file:
        GeneId = 1
        for row in pred_prob:
            if GeneId == 1:
                file.write(b'GeneId,Prediction\n')      
            file.write(str(GeneId).encode() + b',' + str(round(row,4)).encode() + b'\n')
            GeneId = GeneId + 1
    file.close()

