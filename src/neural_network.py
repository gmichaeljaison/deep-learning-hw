import numpy as np
import math
import logging


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
            self.weights[i] = np.empty((self.neurons_cnt[i-1] + 1, self.neurons_cnt[i]))
            self.weights[i][1:, :] = \
                np.random.uniform(-b, b, (self.neurons_cnt[i-1], self.neurons_cnt[i]))
            self.weights[i][0, :] = 0  # bias weights are 0

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
        return hx, ax

    def backward(self, x, y):
        hx, ax = self.forward(x)

        # derivative of ax (reverse order)
        dx = - (y - hx[-1])
        logging.info('loss: {:.4f}'.format(np.absolute(dx).sum()))

        for i in range(self.num_layers-1, 0, -1):
            dw = np.dot(dx, hx[i-1].T)
            dw = np.vstack((dx.T, dw.T))

            # dx = np.vstack(([[1]], dx))
            w_nobias = self.weights[i][1:, :]
            d_hx = np.dot(w_nobias, dx)
            dx = d_hx * sigmoid_grad(ax[i-1])

            # update weights
            self.weights[i] += -self.lr * dw

    def train(self, x_train, y_train, epochs):
        for epoch in range(epochs):
            for i in range(x_train.shape[0]):
                x = x_train[i:i+1, :].T
                y = y_train[i:i+1, :].T
                self.backward(x, y)

    def test(self, x_test):
        y_test_pred = np.empty((x_test.shape[0], self.neurons_cnt[-1]))
        for i in range(x_test.shape[0]):
            y_pred, _ = self.forward(x_test[i:i+1, :].T)
            y_pred = y_pred[-1]
            y_test_pred[i:i+1, :] = y_pred.T
        return y_test_pred


# ------------------
# Module functions #


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    x = np.exp(x)
    x /= x.sum()
    return x
