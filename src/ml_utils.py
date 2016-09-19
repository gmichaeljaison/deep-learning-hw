import numpy as np


def normalize_data(data):
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    return data


def lbl2prob(lbl, num_classes=2):
    y_train = np.zeros((len(lbl), num_classes))
    for i, y in enumerate(lbl):
        y_train[i, int(y)] = 1
    return y_train


def prob2lbl(prob):
    return prob.argmax(axis=1)


def evaluate(y_pred, y):
    return np.count_nonzero(y_pred == y) / len(y)


def shuffle(X, Y):
    rand_i = np.random.permutation(range(len(Y)))
    return X[rand_i], Y[rand_i]
