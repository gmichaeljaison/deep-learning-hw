import math
import logging
import numpy as np

from src.nn import Module
from cv_utils import img_utils


class FullyConnected(Module):
    def __init__(self, name, inp_size, out_size):
        super().__init__(name)
        self.inp_size = inp_size
        self.out_size = out_size

        self.w = FullyConnected.init_weights(inp_size, out_size)
        logging.info('Initializing weights for layer<{}>'.format(self.name))

        self.bias = np.zeros((1, out_size))
        self.bias2 = np.zeros((1, inp_size))

        # Gradients
        self.dw = 0
        self.d_bias = 0
        self.d_bias2 = 0

    def set_weights(self, weights):
        logging.info('Pre-loading weights for layer<{}>'.format(self.name))
        self.w = weights[0]
        self.bias = weights[1] if len(weights) > 1 else 0
        self.bias2 = weights[2] if len(weights) > 2 else 0

    @staticmethod
    def init_weights(inp_size, out_size):
        b = math.sqrt(6) / math.sqrt(out_size + inp_size)
        return np.random.uniform(-b, b, (inp_size, out_size))

    def forward(self, x):
        if x is None:
            return

        super().forward(x)

        self.h = np.dot(x, self.w)
        self.h += self.bias
        return self.h

    def backward(self, h):
        x_cap = np.dot(h, self.w.T)
        x_cap += self.bias2
        return x_cap

    def update_gradient(self, dh):
        """
        Computed gradient for the current layer

        :param x: Input
        :param dh: Gradient of the Output
        :return: dx Gradient of the Input
        """
        bsize = self.x.shape[0]

        self.dw = np.dot(self.x.T, dh) / bsize
        self.d_bias = dh.mean(axis=0)

        self.dx = np.dot(dh, self.w.T)
        return self.dx

    def update_weight(self, lr):
        self.w -= lr * self.dw
        self.bias -= lr * self.d_bias

    def visualize_weight(self):
        imgs = list()
        size = math.sqrt(self.w.shape[0])
        for col in range(self.w.shape[1]):
            w_im = np.reshape(self.w[:, col], (size, size)).astype(np.float32)

            # convert weights range to 0:255
            # http://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
            old_range = w_im.max() - w_im.min()
            w_im = (w_im - w_im.min()) * 255/old_range

            imgs.append(w_im)
        coll = img_utils.collage(imgs, (10, 10))
        return coll
