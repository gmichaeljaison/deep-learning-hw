import numpy as np

from src import ml_utils


class Dataset:
    def __init__(self):
        self.x_train, self.y_train = None, None
        self.x_valid, self.y_valid = None, None
        self.x_test, self.y_test = None, None

    def _apply_transform(self, fn):
        if self.x_train is not None:
            self.x_train, self.y_train = fn(self.x_train, self.y_train)
        if self.x_valid is not None:
            self.x_valid, self.y_valid = fn(self.x_valid, self.y_valid)
        if self.x_test is not None:
            self.x_test, self.y_test = fn(self.x_test, self.y_test)

    @staticmethod
    def _create_xy_fn(fn):
        def new_fn(x, y):
            return fn(x), y
        return new_fn

    def normalize(self):
        """
        :return: Normalized input values. Applied only on X
        """
        self._apply_transform(Dataset._create_xy_fn(ml_utils.normalize_data))

    def binarize(self):
        """
        Round off the values to make the image binary
        """
        self._apply_transform(Dataset._create_xy_fn(np.round))

    def shuffle(self):
        self._apply_transform(ml_utils.shuffle)

    def next_batch(self, batch_size=100):
        n = self.x_train.shape[0]
        n_batches = n // batch_size

        for i in range(n_batches):
            # logging.info('Yielding dataset batch: {}'.format(i+1))
            si = i * batch_size
            end_i = si + batch_size
            x_batch = self.x_train[si:end_i]
            y_batch = self.y_train[si:end_i]
            yield x_batch, y_batch


class MNIST(Dataset):
    def __init__(self, train_fpath, test_fpath=None, valid_fpath=None):
        super().__init__()
        self.x_train, self.y_train = read_input(train_fpath, 10)
        self.x_valid, self.y_valid = read_input(valid_fpath, 10)
        self.x_test, self.y_test = read_input(test_fpath, 10)


def read_input(fname, num_class, prob=True):
    if fname is None:
        return None, None

    data = np.loadtxt(fname, delimiter=',')
    y = data[:, -1]
    x = data[:, :-1]

    if prob:
        y = ml_utils.lbl2prob(y, num_class)

    return x, y
