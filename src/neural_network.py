import numpy as np
import math
import logging

from src import ml_utils


class Net:
    def __init__(self, neurons_cnt, lr=0.1):
        self.neurons_cnt = neurons_cnt
        self.lr = lr

        self.num_layers = len(neurons_cnt)
        self.weights = list()

        self.initialize_weights()

    def initialize_weights(self):
        # no weights for layer0
        self.weights = [None] * self.num_layers

        for i in range(1, self.num_layers):
            b = math.sqrt(6) / math.sqrt(self.neurons_cnt[i] + self.neurons_cnt[i-1])

            self.weights[i] = np.zeros((self.neurons_cnt[i-1] + 1, self.neurons_cnt[i]))
            self.weights[i][1:, :] = \
                np.random.uniform(-b, b, (self.neurons_cnt[i-1], self.neurons_cnt[i]))

    def forward(self, x):
        # pre-activation
        ax = [None] * self.num_layers
        # post-activation
        hx = [None] * self.num_layers

        ax[0] = x
        hx[0] = x

        for i in range(1, self.num_layers):
            hx_bias = np.vstack(([[1]], hx[i-1]))
            ax[i] = np.dot(self.weights[i].T, hx_bias)
            if i < self.num_layers-1:
                hx[i] = sigmoid(ax[i])
            else:
                hx[i] = softmax(ax[i])
        return hx

    def backward(self, x, y):
        hx = self.forward(x)

        # derivative of ax (reverse order)
        dx = - (y - hx[-1])
        # logging.info('loss: {:.4f}'.format(np.absolute(dx).sum()))

        for i in range(self.num_layers-1, 0, -1):
            dw = np.dot(dx, hx[i-1].T)
            dw = np.vstack((dx.T, dw.T))

            # dx = np.vstack(([[1]], dx))
            w_nobias = self.weights[i][1:, :]
            d_hx = np.dot(w_nobias, dx)
            dx = d_hx * (hx[i-1] * (1 - hx[i-1]))  # der(sigmoid)
            # dx = d_hx * sigmoid_prime(ax[i - 1])

            # update weights
            self.weights[i] += -self.lr * dw

    def train(self, x_train, y_train, epochs):
        lbl = ml_utils.prob2lbl(y_train)

        for epoch in range(epochs):
            n_data = x_train.shape[0]
            x_train, y_train = ml_utils.shuffle(x_train, y_train)
            for i in range(n_data):
                x = x_train[i:i+1, :].T
                y = y_train[i:i+1, :].T
                self.backward(x, y)

            if epoch % 100 == 0:
                res = self.test(x_train)
                test_lbl = ml_utils.prob2lbl(res)
                acc = ml_utils.evaluate(test_lbl, lbl)
                logging.info('Epoch: {} - Training-accuracy: {}'.format(epoch, acc))

    def test(self, x_test):
        y_test_pred = np.empty((x_test.shape[0], self.neurons_cnt[-1]))
        for i in range(x_test.shape[0]):
            y_pred = self.forward(x_test[i:i+1, :].T)
            y_pred = y_pred[-1]
            y_test_pred[i:i+1, :] = y_pred.T
        return y_test_pred


# ------------------
# Module functions #


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    x = np.exp(x)
    x /= x.sum()
    return x
