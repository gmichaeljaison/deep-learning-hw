import numpy as np

from src import ml_utils
from src.nn import Module, FullyConnected, Sigmoid


class DBM(Module):
    def __init__(self, name, layers):
        super().__init__(name)

        self.layers = layers
        # initialize bottom and top layers
        bottom = None
        for i, lay in enumerate(self.layers):
            top = self.layers[i+1] if i+1 < len(self.layers) else None
            lay.bottom = bottom
            lay.top = top
            bottom = lay

    @property
    def pcd(self):
        return self.layers[0].h_cap

    def reconstruct(self, x, steps=1):
        # randomly sample h2
        h2 = np.random.randint(0, 2, (x.shape[0], self.layers[2].h_cap.shape[1]))
        # h2 = np.random.uniform(0, 1, (x.shape[0], self.layers[2].h_cap.shape[1]))

        h1_cap, h1_prob = self.layers[1].gibbs_sampling(x, h2, steps)

        # x_prob, _ = self.layers[1].backward(h1_prob)
        x_prob, _ = self.layers[1].backward(h1_cap)
        x_cap = ml_utils.sample_from_prob(x_prob)

        return x_cap, x_prob

    def forward(self, x):
        super().forward(x)

        # mean field updates until convergence
        self.mean_field_update(x)

        # stochastic approximation (persistent CD)
        self.update_samples(1)

    def update_weight(self, lr, momentum=0):
        for lay in self.layers[1:]:
            lay.update_weight(lr)

    def mean_field_update(self, x):
        """
        Variational inference step. Updates the mean of all layers using mean-field update steps.

        :param x: Input image
        """
        self.layers[0].mu = x

        # update until convergence
        max_steps = 5
        for step in range(max_steps):
            if step == 0:
                for lay in self.layers[1:]:
                    lay.mu = np.random.uniform(0, 1, (x.shape[0], lay.fc.out_size))

            dif = 0
            for lay in self.layers[1:]:
                prev_mu = lay.mu
                lay.mean_field_update_step()
                dif += abs(prev_mu - lay.mu).sum()

    def update_samples(self, steps=1):
        bottom = None

        for i, lay in enumerate(self.layers):
            top = self.layers[i+1].h_cap if i+1 < len(self.layers) else None

            h_cap_prev = lay.h_cap  # h_cap in iteration 't'
            lay.h_cap, _ = lay.gibbs_sampling(bottom, top, steps)  # iteration 't+1'
            # bottom = h_cap_prev
            bottom = lay.h_cap


class DBMLayer0:
    def __init__(self, name, x_size, n_chains, top=None):
        self.name = name
        self.mu = None
        self.h_cap = np.random.randint(0, 2, (n_chains, x_size))
        self.bottom = None
        self.top = top

    def gibbs_sampling(self, bottom, top, steps=1):
        if top is None:
            return

        # step 1
        v_prob, h2_prob = self.top.backward(top)
        v_cap = ml_utils.sample_from_prob(v_prob)
        h2_cap = ml_utils.sample_from_prob(h2_prob)

        # step 2 to k
        for step in range(1, steps):
            top_prob = self.top.forward((v_cap, h2_cap))
            top = ml_utils.sample_from_prob(top_prob)

            v_prob, h2_prob = self.top.backward(top)
            v_cap = ml_utils.sample_from_prob(v_prob)
            h2_cap = ml_utils.sample_from_prob(h2_prob)

        return v_cap, v_prob


class DBMLayer(Module):
    def __init__(self, name, inp_size, out_size, n_chains, bottom=None, top=None):
        super().__init__(name)

        self.fc = FullyConnected('{}-fc'.format(name), inp_size, out_size)
        self.activation = Sigmoid('sigm')

        # reference to top & bottom layer
        self.top = top
        self.bottom = bottom

        # mean
        self.mu = None

        # MCMC sampled h using Gibbs sampling at iteration 't'. Initialized randomly.
        self.h_cap = np.random.randint(0, 2, (n_chains, out_size))

    def forward(self, x):
        """
        Samples h given the output from previous layer and backward output from next layer

        :param x: Tuple of (i-1 layer value, i+1 layer value)
        :return: h = sigmoid(forward(x[0]) + x[1])
        """
        bottom, top = x
        if bottom is None and top is None:
            return

        h1, h2 = 0, 0
        if bottom is not None:
            h1 = self.fc.forward(bottom)

        if top is not None:
            h2 = self.top.fc.backward(top)

        h = self.activation.forward(h1 + h2)
        return h

    def backward(self, h):
        bottom, top = None, None
        if self.top:
            top = self.activation.forward(self.top.fc.forward(h))
        bottom = self.activation.forward(self.fc.backward(h))
        return bottom, top

    def mean_field_update_step(self):
        mu_bottom = self.bottom.mu if self.bottom else None
        mu_top = self.top.mu if self.top else None
        self.mu = self.forward((mu_bottom, mu_top))

    def gibbs_sampling(self, bottom, top, steps=1):
        if top is None and bottom is None:
            return

        # step1
        h_prob = self.forward((bottom, top))
        h_cap = ml_utils.sample_from_prob(h_prob)

        # step2 to K
        for _ in range(1, steps):
            # prepare for next step
            bottom_prob, top_prob = self.backward(h_cap)
            if bottom_prob is not None:
                bottom = ml_utils.sample_from_prob(bottom_prob)
            if top_prob is not None:
                top = ml_utils.sample_from_prob(top_prob)

            h_prob = self.forward((bottom, top))
            h_cap = ml_utils.sample_from_prob(h_prob)

        return h_cap, h_prob

    def update_weight(self, lr, momentum=0):
        bsize = self.mu.shape[0]
        n_chain = self.h_cap.shape[0]

        dw1 = np.dot(self.bottom.mu.T, self.mu) / bsize
        dw2 = np.dot(self.bottom.h_cap.T, self.h_cap) / n_chain
        dw = dw1 - dw2
        self.fc.dw = (momentum * self.fc.dw) + dw
        self.fc.w += lr * self.fc.dw

        self.fc.d_bias = (momentum * self.fc.d_bias) + \
                         (self.mu.mean(axis=0) - self.h_cap.mean(axis=0))
        self.fc.bias += lr * self.fc.d_bias

        self.fc.d_bias2 = (momentum * self.fc.d_bias2) + \
                          (self.bottom.mu.mean(axis=0) - self.bottom.h_cap.mean(axis=0))
        self.fc.bias2 += lr * self.fc.d_bias2
