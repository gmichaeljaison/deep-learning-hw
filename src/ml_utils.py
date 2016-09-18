import numpy as np


def normalize_data(data):
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    return data


def lbl2prob(lbl):
    y_train = np.zeros((len(lbl), 2))
    for i, y in enumerate(lbl):
        y_train[i, int(y)] = 1
    return y_train
