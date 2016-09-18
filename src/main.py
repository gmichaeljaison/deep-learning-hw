import numpy as np
import logging

from cv_utils import img_utils
from src.neural_network import Net
from src import ml_utils


def read_input(fname):
    data = np.loadtxt(fname, delimiter=',')
    lbl = data[:, -1]
    data = data[:, :-1]
    return data, lbl


def disp_image(vec):
    img = np.reshape(vec, (28, 28))
    img_utils.imshow(img)


def main():
    # data, lbl = read_input('../data/digitsvalid.txt')
    # print(data.shape, data[0, 0])
    # print(lbl.shape, lbl[0])
    # disp_image(data[300, :])

    data, lbl = read_input('../data/xor_train.txt')
    print(data)
    print(lbl)

    net = Net([2, 10, 10, 2], 0.05)

    data = ml_utils.normalize_data(data)
    y_train = ml_utils.lbl2prob(lbl)

    print(data)
    print(y_train)

    net.train(data, y_train, 500)
    res = net.test(data)
    print(res)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
