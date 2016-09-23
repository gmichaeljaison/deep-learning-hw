import numpy as np


def normalize_data(data):
    data -= data.mean(axis=0)
    data /= data.std(axis=0) + 1e-5
    return data


def lbl2prob(lbl, num_classes=2):
    y_train = np.zeros((len(lbl), num_classes))
    for i, y in enumerate(lbl):
        y_train[i, int(y)] = 1
    return y_train


def prob2lbl(prob):
    return prob.argmax(axis=1)


def evaluate(lbl_pred, lbl):
    return np.count_nonzero(lbl_pred == lbl) / len(lbl)


def cross_entropy_error(y_pred, y):
    # loss function (cross-entropy loss)
    loss = np.sum(-np.log(np.sum(y_pred * y, axis=1)))
    loss /= len(y)
    return loss


def classification_error(y_pred, y):
    loss = 1 - evaluate(prob2lbl(y_pred), prob2lbl(y))
    return loss * 100


def shuffle(X, Y):
    rand_i = np.random.permutation(range(len(Y)))
    return X[rand_i], Y[rand_i]
