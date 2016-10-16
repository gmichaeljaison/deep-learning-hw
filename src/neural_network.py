import numpy as np
import math
import logging
import pickle as pkl

from src import ml_utils
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
        self.weight_grads = [None] * self.num_layers  # gradient of previous iteration

        self.initialize_weights()

    @classmethod
    def load(cls, fpath):
        with open(fpath, 'rb') as f:
            return pkl.load(f)

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
        """
        Used for test phase
        """
        hx = x
        for i in range(1, self.num_layers):
            ones = np.ones((hx.shape[0], 1))
            hx = np.hstack((ones, hx))

            ax = np.dot(hx, self.weights[i])
            if i < self.num_layers - 1:
                hx = sigmoid(ax)
                # dropout (test time)
                hx[i] *= (1 - self.dropout)
            else:
                hx = softmax(ax)
        return hx

    def forward(self, x):
        """
        Used for training
        """
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

                # dropout
                mask = np.ones(hx[i].size)
                indices = np.random.permutation(hx[i].size)
                dropout_cnt = int(hx[i].size * self.dropout)
                indices = indices[:dropout_cnt]
                mask[indices] = 0
                hx[i] *= mask.reshape(hx[i].shape)
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

            # Regularization
            dw[1:] += self.weight_decay * regularizer_grad(dw[1:])

            # Momentum
            if self.weight_grads[i] is not None:
                dw += self.momentum * self.weight_grads[i]  # dW(t) = dW(t) + beta * dW(t-1)
            self.weight_grads[i] = dw

            d_hx = np.dot(dx, self.weights[i].T)
            dx = d_hx * (hx[i-1] * (1 - hx[i-1]))  # der(sigmoid)
            dx = dx[:, 1:]

            # update weights
            self.weights[i] -= self.lr * dw

    def train(self, x_train, y_train, x_valid, y_valid, epochs, batch_size=100):
        x_train, y_train = ml_utils.shuffle(x_train, y_train)
        n_data = len(y_train)

        plt_x = list()
        err_train, err_valid = list(), list()
        cerr_train, cerr_valid = list(), list()

        # min error on validation set
        min_err = 100
        min_cerr = 100

        for epoch in range(1, epochs+1):
            for start_i in range(0, n_data, batch_size):
                end_i = start_i + batch_size
                x_batch = x_train[start_i:end_i, :]
                y_batch = y_train[start_i:end_i, :]

                self.backward(x_batch, y_batch)

            if epoch % 20 == 0 or epoch == 1:
                plt_x.append(epoch)
                logging.info('\n\nEpoch: {}'.format(epoch))

                err, cerr = self.validate(x_train, y_train)
                err_train.append(err)
                cerr_train.append(cerr)
                logging.info('Training: Cross-entropy Error: {:.3f}, Classification-error: {:.3f}'.
                             format(err, cerr))

                err, cerr = self.validate(x_valid, y_valid)
                err_valid.append(err)
                cerr_valid.append(cerr)
                logging.info('Validation: Cross-entropy Error: {:.3f}, Classification-error: {:.3f}'.
                             format(err, cerr))
                if err < min_err:
                    min_err = err
                if cerr < min_cerr:
                    min_cerr = cerr

        return plt_x, (err_train, cerr_train), (err_valid, cerr_valid)

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

    def save(self, fpath):
        with open(fpath, 'wb') as f:
            pkl.dump(self, f)


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
