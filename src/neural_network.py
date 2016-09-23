import numpy as np
import math
import logging

from src import ml_utils
from matplotlib import pyplot as plt
from cv_utils import img_utils


class Net:
    def __init__(self, neurons_cnt, lr=0.1, momentum=0, weight_decay=0.5, dropout=0):
        self.neurons_cnt = neurons_cnt
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dropout = dropout

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
            logging.info('Initializing weights for layer<{}> in range {} to {}'.format(i, -b, b))

    def predict(self, x):
        hx = x
        for i in range(1, self.num_layers):
            ones = np.ones((hx.shape[0], 1))
            hx = np.hstack((ones, hx))

            ax = np.dot(hx, self.weights[i])
            if i < self.num_layers - 1:
                hx = sigmoid(ax)
            else:
                hx = softmax(ax)
        return hx

    def forward(self, x):
        bsize = x.shape[0]

        # post-activation
        hx = [None] * self.num_layers
        hx[0] = x

        for i in range(1, self.num_layers):
            ones = np.ones((bsize, 1))
            hx[i-1] = np.hstack((ones, hx[i-1]))

            ax = np.dot(hx[i-1], self.weights[i])
            if i < self.num_layers-1:
                hx[i] = sigmoid(ax)
            else:
                hx[i] = softmax(ax)
        return hx

    def backward(self, x, y):
        bsize = x.shape[0]
        hx = self.forward(x)

        # derivative of ax (reverse order)
        dx = - (y - hx[-1])

        for i in range(self.num_layers-1, 0, -1):
            dw = np.dot(hx[i-1].T, dx)
            dw /= bsize
            dw[1:] += self.weight_decay * regularizer_grad(dw[1:])

            d_hx = np.dot(dx, self.weights[i].T)
            dx = d_hx * (hx[i-1] * (1 - hx[i-1]))  # der(sigmoid)
            dx = dx[:, 1:]

            # update weights
            self.weights[i] -= self.lr * dw

    def train(self, x_train, y_train, x_valid, y_valid, epochs, batch_size=100, visualize=False):
        x_train, y_train = ml_utils.shuffle(x_train, y_train)
        n_data = len(y_train)

        plt_x = list()
        err_train, err_valid = list(), list()
        cerr_train, cerr_valid = list(), list()

        for epoch in range(epochs):

            for start_i in range(0, n_data, batch_size):
                end_i = start_i + batch_size
                x_batch = x_train[start_i:end_i, :]
                y_batch = y_train[start_i:end_i, :]

                self.backward(x_batch, y_batch)

            if epoch % 10 == 0:
                plt_x.append(epoch)
                logging.info('\n\nEpoch: {}'.format(epoch))

                err, cerr = self.validate(x_train, y_train)
                err_train.append(err)
                cerr_train.append(cerr)
                logging.info('Training - Cross-entropy Error: {:.3f}, Classification-error: {:.3f}'.
                             format(err, cerr))

                err, cerr = self.validate(x_valid, y_valid)
                err_valid.append(err)
                cerr_valid.append(cerr)
                logging.info('Validation - Cross-entropy Error: {:.3f}, Classification-error: {:.3f}'.
                             format(err, cerr))

        if visualize:
            plt.plot(plt_x, err_train, 'bs', plt_x, err_valid, 'g^')
            plt.ylabel('Cross-entropy error')
            plt.xlabel('Epochs')
            plt.ylim((0, 1))
            plt.show()

            plt.figure()
            plt.plot(plt_x, cerr_train, 'bs', plt_x, cerr_valid, 'g^')
            plt.ylabel('Classification error')
            plt.xlabel('Epochs')
            plt.show()

    def validate(self, x, y):
        res = self.predict(x)
        err = ml_utils.cross_entropy_error(res, y)
        cerr = ml_utils.classification_error(res, y)
        return err, cerr

    def visualize_weights(self, size, layer=1):
        w = self.weights[layer]
        imgs = list()
        for col in range(w.shape[1]):
            w_im = np.reshape(w[1:, col], size)
            imgs.append(w_im.astype(np.float32) * 255)
        coll = img_utils.collage(imgs, (10, 10))
        return coll


# ------------------
# Module functions #


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    x = np.exp(x)
    x /= np.atleast_2d(x.sum(axis=1)).T
    return x


def regularizer_grad(w):
    return 2 * w
