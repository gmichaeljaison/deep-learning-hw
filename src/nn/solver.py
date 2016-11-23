import os.path as osp
import logging
import pickle as pkl

from src import ml_utils


class Solver:
    def __init__(self, net, lr, epochs, batch_size=100, momentum=0, weight_decay=0,
                 test_interval=20):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs if isinstance(epochs, (list, tuple)) else (1, epochs)

        self.batch_size = batch_size
        self.test_interval = test_interval

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

    def save_model(self, epoch, save_dir, save_freq=-1):
        if epoch > 0 and save_freq > 0:
            if epoch == self.epochs[1] or epoch % save_freq == 0:
                fname = '{}-ep{:03d}.pkl'.format(self.net.name, epoch)
                with open(osp.join(save_dir, fname), 'wb') as f:
                    pkl.dump(self.net, f)
                    logging.info('Saving the model at {}'.format(fname))


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
