import numpy as np

from src.nn import Module


class Sigmoid(Module):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, x):
        super(Sigmoid, self).forward(x)

        self.h = 1 / (1 + np.exp(-x))
        return self.h

    def update_gradient(self, dh):
        self.dx = dh * self.h * (1 - self.h)
        return self.dx

    def update_weight(self, lr, momentum=0):
        pass
    
    def der_x_entropy(self, y):
        """
        Derivative of cross-entropy loss function with respect to the output

        x_entropy_loss = - (y * np.log(h) + (1 - y) * np.log(1 - h))
        d(loss)/dw = - [ (y / h) - ((1-y) / (1 - h)) ]
                   = (h - y) / (h * (1 - h))
        """
        dh = (self.h - y) / (self.h * (1 - self.h))
        return dh


class SoftMax(Module):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, x):
        super().forward(x)

        self.h = np.exp(x)
        self.h /= np.atleast_2d(self.h.sum(axis=1)).T
        return self.h

    def update_gradient(self, dh):
        self.dx = - (dh - self.h)
        return self.dx

    def update_weight(self, lr, momentum=0):
        pass

    def der_x_entropy(self, y):
        return y
