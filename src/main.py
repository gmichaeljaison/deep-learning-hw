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
    x_train, lbl = read_input('../data/digitsvalid.txt')
    # disp_image(data[300, :])
    net = Net([784, 100, 10], 0.2)
    x_train = ml_utils.normalize_data(x_train)
    y_train = ml_utils.lbl2prob(lbl, 10)

    # xor - data
    # data, lbl = read_input('../data/xor_train.txt')
    # print(data)
    # print(lbl)
    # net = Net([2, 3, 2], 0.1)
    # y_train = ml_utils.lbl2prob(lbl, 2)
    # print(data)
    # print(y_train)

    net.train(x_train, y_train, 2000)

    res = net.test(x_train)
    test_lbl = ml_utils.prob2lbl(res)
    print(ml_utils.evaluate(test_lbl, lbl))

    # print(res)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
