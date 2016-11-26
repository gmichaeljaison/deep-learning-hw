import os.path as osp
import logging
import pickle as pkl

from src import ml_utils
from cv_utils import utils


class Solver:
    def __init__(self, net, lr, epochs, batch_size=100, momentum=0, weight_decay=0,
                 test_interval=20):
        logging.info('Initialzing the solver\n'
                     'Learning rate: {}\n'
                     'Epochs: {}\n'
                     'batch_size: {}\n'
                     'Momentum: {}\n'
                     'Weight decay: {}\n'.format(lr, epochs, batch_size, momentum, weight_decay))
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs if isinstance(epochs, (list, tuple)) else (1, epochs)

        self.batch_size = batch_size
        self.test_interval = test_interval

        self._save_dir = net.name

        # set of functions to execute as a pipeline during the training process
        # format is tuple: (function, epoch) -> function to execute at every <epoch>s
        self.pipeline = list()

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, save_dir):
        self._save_dir = osp.join(save_dir, self.net.name)
        utils.create_dirs(self._save_dir)

    def run_pipeline(self, epoch):
        for fn, every_ep in self.pipeline:
            if epoch % every_ep == 0:
                logging.info('Invoking {} at epoch {}'.format(fn.__name__, epoch))
                fn(self, epoch)

    def add_pipe(self, fn, freq, pos=-1):
        if pos >= 0:
            self.pipeline.insert(pos, (fn, freq))
        else:
            self.pipeline.append((fn, freq))

    def validate(self, x, y):
        res = self.net.forward(x)
        err = ml_utils.cross_entropy_error(res, y)
        cerr = ml_utils.classification_error(res, y)
        return err, cerr

    def solve(self, dataset):
        n_data = dataset.x_train.shape[0]

        for epoch in range(self.epochs[0], self.epochs[1]+1):
            for si in range(0, n_data, self.batch_size):
                x_batch = dataset.x_train[si:si + self.batch_size]
                y_batch = dataset.y_train[si:si + self.batch_size]

                self.net.forward(x_batch)
                dh = self.net.output_gradient(y_batch)
                self.net.back_propagate(dh, self.lr)

            yield epoch + 1

    def train(self, dataset):
        errs = list()
        for epoch in self.solve(dataset):
            if epoch % self.test_interval == 0:
                train_err = self.validate(dataset.x_train, dataset.y_train)
                valid_err = self.validate(dataset.x_valid, dataset.y_valid)

                logging.info('Training error. Epoch: {}, Cross-entropy error: {}, '
                             'Classification error: {}'.format(epoch, train_err[0], train_err[1]))
                logging.info('Validation error. Epoch: {}, Cross-entropy error: {}, '
                             'Classification error: {}'.format(epoch, valid_err[0], valid_err[1]))
                errs.append((epoch, train_err, valid_err))

        return errs

    def add_save_pipe(self, save_dir, save_freq):
        self.save_dir = save_dir
        self.add_pipe(Solver.save_model, save_freq, 0)

    def save_model(self, epoch):
        fname = '{}-ep{:03d}.pkl'.format(self.net.name, epoch)
        with open(osp.join(self.save_dir, fname), 'wb') as f:
            pkl.dump(self.net, f)
        logging.info('Saving the model at {}'.format(fname))

    @staticmethod
    def load_model(name, epoch, save_dir=''):
        save_dir = osp.join(save_dir, name)
        fname = '{}-ep{:03d}.pkl'.format(name, epoch)
        with open(osp.join(save_dir, fname), 'rb') as f:
            return pkl.load(f)


class AutoEncoderSolver(Solver):
    def __init__(self, net, lr, epochs, batch_size=100, momentum=0, weight_decay=0,
                 test_interval=20):
        super().__init__(net, lr, epochs, batch_size, momentum, weight_decay, test_interval)

    def validate(self, x, y):
        x_cap = self.net.forward(x)
        err = ml_utils.cross_entropy_reconstruction_error(x, x_cap)
        return err

    def solve(self, dataset):
        n_data = dataset.x_train.shape[0]

        for epoch in range(self.epochs[0], self.epochs[1]+1):
            for si in range(0, n_data, self.batch_size):
                x_batch = dataset.x_train[si:si + self.batch_size]

                self.net.forward(x_batch)
                dh = self.net.output_gradient(x_batch)
                self.net.back_propagate(dh, self.lr)

            yield epoch + 1

    def train(self, dataset):
        errs = list()
        for epoch in self.solve(dataset):
            if epoch % self.test_interval == 0:
                train_err = self.validate(dataset.x_train, dataset.y_train)
                valid_err = self.validate(dataset.x_valid, dataset.y_valid)

                logging.info('Training error. Epoch: {}, Cross-entropy error: {}'
                             .format(epoch, train_err))
                logging.info('Validation error. Epoch: {}, Cross-entropy error: {}'
                             .format(epoch, valid_err))
                errs.append((epoch, train_err, valid_err))

        return errs


class GenSolver(Solver):
    """
    Solver for unsupervised generative models
    1. Restricted Boltzmann Machine
    2. Deep Boltzmann Machine
    """
    def __init__(self, net, lr, epochs, batch_size=100, momentum=0, test_interval=20):
        super().__init__(net, lr, epochs, batch_size, momentum,
                         test_interval=test_interval)

    def validate(self, x, y):
        _, x_prob = self.net.reconstruct(x, 1)
        err = ml_utils.cross_entropy_reconstruction_error(x, x_prob)
        return err

    def train(self, dataset):
        plt_x, terrs, verrs = list(), list(), list()

        def train_val_err(solver, epoch):
            plt_x.append(epoch)

            terr = solver.validate(dataset.x_train, None)
            logging.info('Training Cross-entropy error: {}'.format(terr))
            terrs.append(terr)

            verr = solver.validate(dataset.x_valid, None)
            logging.info('Validation Cross-entropy error: {}'.format(verr))
            verrs.append(verr)

        self.add_pipe(train_val_err, freq=self.test_interval, pos=0)
        train_val_err(self, self.epochs[0]-1)

        for epoch in range(self.epochs[0], self.epochs[1]+1):
            logging.info('starting epoch {}'.format(epoch))

            for x_batch, _ in dataset.next_batch(self.batch_size):
                self.net.forward(x_batch)
                self.net.update_weight(self.lr, self.momentum)

            self.run_pipeline(epoch)
        return plt_x, terrs, verrs
