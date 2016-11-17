import numpy as np

from src import ml_utils


class Dataset:
    def __init__(self):
        self.x_train, self.y_train = None, None
        self.x_valid, self.y_valid = None, None
        self.x_test, self.y_test = None, None

    def normalize(self):
        self.x_train = ml_utils.normalize_data(self.x_train)
        self.x_valid = ml_utils.normalize_data(self.x_valid)
        self.x_test = ml_utils.normalize_data(self.x_test)

    def binarize(self):
        """
        Round off the values to make the image binary
        """
        self.x_train = np.round(self.x_train)
        self.x_valid = np.round(self.x_valid)
        self.x_test = np.round(self.x_test)

    def shuffle(self):
        self.x_train, self.y_train = ml_utils.shuffle(self.x_train, self.y_train)
        self.x_valid, self.y_valid = ml_utils.shuffle(self.x_valid, self.y_valid)
        self.x_test, self.y_test = ml_utils.shuffle(self.x_test, self.y_test)
        pass


class MNIST(Dataset):
    def __init__(self, train_fpath, test_fpath, valid_fpath):
        super().__init__()
        self.x_train, self.y_train = read_input(train_fpath, 10)
        self.x_valid, self.y_valid = read_input(valid_fpath, 10)
        self.x_test, self.y_test = read_input(test_fpath, 10)


def read_input(fname, num_class, prob=True):
    data = np.loadtxt(fname, delimiter=',')
    y = data[:, -1]
    x = data[:, :-1]

    if prob:
        y = ml_utils.lbl2prob(y, num_class)

    return x, y
