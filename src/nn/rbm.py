import numpy as np

from src.nn import Module, FullyConnected, Sigmoid


class RBM(Module):
    def __init__(self, name, inp_size, out_size):
        super().__init__(name)

        self.fc = FullyConnected('fc', inp_size, out_size)
        self.activation = Sigmoid('sigm')
        self.train_steps = 1

        # Weight params
        self.bias2 = np.zeros((1, inp_size))

        # Gradients
        self.d_bias2 = None

    def apply_forward(self, x):
        ax = self.fc.forward(x)
        h = self.activation.forward(ax)
        h = h.round()
        return h

    def forward(self, x):
        super().forward(x)
        self.h = self.apply_forward(x)
        return self.h

    def backward(self, h):
        x_cap = np.dot(h, self.fc.w.T) + self.bias2
        x_cap = self.activation.forward(x_cap)
        return x_cap

    def update_weight(self, lr):
        x_cap, h_cap = self.gibbs_sampling(self.x, self.train_steps)

        self.fc.dw = (np.dot(self.x.T, self.h) - np.dot(x_cap.T, h_cap)) / self.x.shape[0]
        self.fc.d_bias = (self.h - h_cap).mean(axis=0)
        self.d_bias2 = (self.x - x_cap).mean(axis=0)

        self.bias2 += lr * self.d_bias2
        self.fc.bias += lr * self.fc.d_bias
        self.fc.w += lr * self.fc.dw

    def update_gradient(self, dh):
        pass

    def gibbs_sampling(self, x, steps=1):
        x_cap = x

        prob_h = self.apply_forward(x_cap)
        h_cap = prob_h.round()

        def pick_prob(mat):
            """ Pick 0 or 1 based on the probability values """
            r = np.random.uniform(0, 1, mat.shape)
            return (mat > r).astype(int)

        for _ in range(steps):
            prob_x = self.backward(h_cap)
            x_cap = pick_prob(prob_x)

            prob_h = self.apply_forward(x_cap)
            h_cap = pick_prob(prob_h)

        return x_cap, h_cap


def solver(rbm, dataset, epochs, lr=0.1, bsize=100):
    """
    Trains the network using Restricted Boltzmann Machine

    :param rbm: RBM Module
    :param dataset: Dataset object
    :param epochs: Number of epochs to iterate
    :param lr: Learning rate
    :param bsize: Batch size
    :yield: Yields loss information every epoch
    """
    n_data = dataset.x_train.shape[0]
    for epoch in range(epochs):
        for si in range(0, n_data, bsize):
            x_batch = dataset.x_train[si:si + bsize]

            rbm.forward(x_batch)
            rbm.update_weight(lr)

        yield epoch + 1
