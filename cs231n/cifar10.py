'''
This is a bunch of utility routines for handling the CIFAR-10 dataset
in Python3.
Author: Jamie Hall
License: GPL2
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler


def get_batch_data(batch_name):
    file_name = os.path.join("input", batch_name)
    with open(file_name, "rb") as file:
        # Encoding kwarg needed for reading a Py2 pickled object
        data_dict = pickle.load(file, encoding="bytes")
    X = data_dict[b"data"]
    y = data_dict[b"labels"]
    return X, y


def assemble_image_data():
    X_train, y_train = get_batch_data("data_batch_1")
    for i in range(2,6):
        X0, y0 = get_batch_data("data_batch_" + str(i))
        X_train = np.vstack((X_train,X0))
        y_train = np.concatenate((y_train, y0))
    X_test, y_test = get_batch_data("test_batch")
    return X_train, y_train, X_test, y_test


def normalise_image_data(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def plot_image(image_row_data, file_name):
    int_data = np.array(image_row_data, np.uint8)
    # 'order' kwarg respects the Fortranish row-major ordering of the CIFAR data;
    # the transpose() flips the image 90deg clockwise.
    im = np.reshape(int_data, (32,32,3), order="F").transpose((1,0,2))
    plt.figure()
    plt.axis("off")
    plt.imshow(im)
    plt.savefig(file_name)


def get_normalised_data():
    xtrain, y_train, xtest, y_test = assemble_image_data()
    X_train, X_test, scaler = normalise_image_data(xtrain, xtest)
    return X_train, y_train, X_test, y_test, scaler
