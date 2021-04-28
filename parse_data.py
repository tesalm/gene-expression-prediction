# -*- coding: utf-8 -*-

import numpy as np
import os


def load_data():
    data_path = "data" # This folder holds the csv files

    # Load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.

    print("Loading data...")
    x_train = np.loadtxt(data_path + os.sep + "x_train.csv", 
                         delimiter = ",", skiprows = 1)
    x_test  = np.loadtxt(data_path + os.sep + "x_test.csv", 
                         delimiter = ",", skiprows = 1)    
    y_train = np.loadtxt(data_path + os.sep + "y_train.csv", 
                         delimiter = ",", skiprows = 1)

    print("All files loaded. Preprocessing...")

    # Remove the first column(Id)
    x_train = x_train[:,1:]
    x_test  = x_test[:,1:]
    y_train = y_train[:,1:]

    # Every 100 rows correspond to one gene.
    # Extract all 100-row-blocks into a list using np.split.
    num_genes_train = x_train.shape[0] / 100
    num_genes_test  = x_test.shape[0] / 100

    print("Train / test data has %d / %d genes." % \
          (num_genes_train, num_genes_test))
    x_train = np.split(x_train, num_genes_train)
    x_test  = np.split(x_test, num_genes_test)

    return x_train, y_train, x_test


def preprocess_data():
    x_train, y_train, x_test = load_data()

    # Reshape by raveling each 100x5 array into a 500-length vector
    x_train = [g.ravel() for g in x_train]
    x_test  = [g.ravel() for g in x_test]

    # Convert data from list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_train = np.ravel(y_train)

    # Now x_train should be 15485 x 500 and x_test 3871 x 500.
    # y_train is 15485-long vector.

    print("x_train shape is %s" % str(x_train.shape))    
    print("y_train shape is %s" % str(y_train.shape))
    print("x_test shape is %s" % str(x_test.shape))

    print('Data preprocessing done...')
    print("-" * 40 + '\n')

    return x_train, y_train, x_test


def preprocess_data_for_cnn():
    x_train_cnn = []; x_test_cnn = []
    x_train, y_train, x_test = load_data()

    # Transform x_train dataset from shape (15485L, 100, 5) to (15485L, 5, 100).
    for i in x_train:
        x_train_cnn.append(np.transpose(i))

    # Transform x_test dataset from shape (3871L, 100, 5) to (3871L, 5, 100).
    for i in x_test:
        x_test_cnn.append(np.transpose(i))

    # Convert data from list to array
    x_train_cnn = np.array(x_train_cnn)
    x_test_cnn = np.array(x_test_cnn)
    y_train = np.array(y_train)
    y_train = np.ravel(y_train)

    print("x_train_cnn shape is %s" % str(x_train_cnn.shape))    
    print("y_train shape is %s" % str(y_train.shape))
    print("x_test_cnn shape is %s" % str(x_test_cnn.shape))

    print('Data preprocessing done...')
    print("-" * 40 + '\n')

    return x_train_cnn, y_train, x_test_cnn


