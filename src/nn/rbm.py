import numpy as np

from src import ml_utils
from src.nn import Module, FullyConnected, Sigmoid


class RBM(Module):
    def __init__(self, name, inp_size, out_size, n_chain):
        super().__init__(name)

        self.fc = FullyConnected('{}-fc'.format(name), inp_size, out_size)
        self.activation = Sigmoid('{}-sigm'.format(name))
        self.train_steps = 1

        # Weight params
        self.bias2 = np.zeros((1, inp_size))

        # Gradients
        self.d_bias2 = np.zeros((1, inp_size))

        # Persistence CD chains
        self.x_cap = np.random.randint(0, 2, (n_chain, inp_size))

    @property
    def pcd(self):
        return self.x_cap

    def reconstruct(self, x, steps=1):
        _, h_cap = self.gibbs_sampling(x, steps)

        x_prob = self.backward(h_cap)
        x_cap = ml_utils.sample_from_prob(x_prob)
        return x_cap, x_prob

    def apply_forward(self, x):
        ax = self.fc.forward(x)
        h = self.activation.forward(ax)
        # h = h.round()
        return h

    def forward(self, x):
        super().forward(x)
        self.h = self.apply_forward(x)
        return self.h

    def backward(self, h):
        x_cap = np.dot(h, self.fc.w.T) + self.bias2
        x_cap = self.activation.forward(x_cap)
        return x_cap

    def update_weight(self, lr, momentum=0):
        bsize = self.x.shape[0]
        n_chain = self.x_cap.shape[0]

        self.x_cap, h_cap = self.gibbs_sampling(self.x_cap, self.train_steps)

        dw1 = np.dot(self.x.T, self.h) / bsize
        dw2 = np.dot(self.x_cap.T, h_cap) / n_chain
        dw = dw1 - dw2
        self.fc.dw = (momentum * self.fc.dw) + dw
        self.fc.w += lr * self.fc.dw
        # self.fc.dw = (np.dot(self.x.T, self.h) - np.dot(self.x_cap.T, h_cap)) / self.x.shape[0]

        self.fc.d_bias = (momentum * self.fc.d_bias) + \
                         (self.h.mean(axis=0) - h_cap.mean(axis=0))
        self.fc.bias += lr * self.fc.d_bias

        self.d_bias2 = (momentum * self.d_bias2) + \
                       (self.x.mean(axis=0) - self.x_cap.mean(axis=0))
        self.bias2 += lr * self.d_bias2

    def update_gradient(self, dh):
        pass

    def gibbs_sampling(self, x, steps=1):
        x_cap, h_cap = x, None

        for _ in range(steps):
            h_prob = self.apply_forward(x_cap)
            h_cap = ml_utils.sample_from_prob(h_prob)

            x_prob = self.backward(h_cap)
            x_cap = ml_utils.sample_from_prob(x_prob)

        return x_cap, h_cap
