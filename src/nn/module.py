import numpy as np
import pickle as pkl


class Module:
    def __init__(self, name):
        self.name = name

        self.x = None  # Input
        self.h = None  # Output

        self.dx = None  # Input gradient
        self.dh = None  # Output gradient

    @classmethod
    def load(cls, fpath):
        with open(fpath, 'rb') as f:
            return pkl.load(f)

    def forward(self, x):
        """
        Forward propagate and compute the output for this Module

        :param x: Input
        :return: Output
        """
        self.x = x

    def back_propagate(self, dh, lr):
        self.dh = dh
        self.update_gradient(dh)
        self.update_weight(lr)

    def update_gradient(self, dh):
        """
        Update the input gradient (dx)

        :param dh: Gradient of the output
        :return:
        """
        raise NotImplementedError('Layer is an interface. This method needs an implementation')

    def update_weight(self, lr, momentum=0):
        raise NotImplementedError('Layer is an interface. This method needs an implementation')

    def save(self, fpath):
        with open(fpath, 'wb') as f:
            pkl.dump(self, f)


class Dropout(Module):
    def __init__(self, name, percent):
        super().__init__(name)

        self.percent = percent
        self.train = True
        self.mask = None

    def forward(self, x):
        super().forward(x)

        if self.train:
            self.mask = np.random.choice(2, x.shape, p=[self.percent, 1-self.percent])
            self.h = x * self.mask
        else:
            self.h = x * (1 - self.percent)
        return self.h

    def update_gradient(self, dh):
        self.dh = dh
        self.dh *= self.mask
        return self.dh

    def update_weight(self, lr, momentum=0):
        pass
