import numpy as np
import logging
import sys

# from cv_utils import img_utils
from src.neural_network import Net
from src import ml_utils
from matplotlib import pyplot as plt


def read_input(fname, num_class, normalize=False, prob=True):
    data = np.loadtxt(fname, delimiter=',')
    y = data[:, -1]
    x = data[:, :-1]

    if normalize:
        x = ml_utils.normalize_data(x)

    if prob:
        y = ml_utils.lbl2prob(y, num_class)

    return x, y


# def disp_image(vec):
#     img = np.reshape(vec, (28, 28))
#     img_utils.imshow(img)


def plot_err(plt_x, err_train, err_valid):
    plt.plot(plt_x, err_train[0], 'bs', label='Training error')
    plt.plot(plt_x, err_valid[0], 'g^', label='Validation error')
    plt.ylabel('Cross-entropy error')
    plt.xlabel('Epochs')
    plt.ylim((0, 1))
    plt.legend()
    plt.show()
    plt.close('all')

    plt.figure()
    plt.plot(plt_x, err_train[1], 'bs', label='Training error')
    plt.plot(plt_x, err_valid[1], 'g^', label='Validation error')
    plt.ylabel('Classification error')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.close('all')


def train(net, epochs=500):
    x_train, y_train = read_input('../data/digitstrain.txt', 10)
    x_valid, y_valid = read_input('../data/digitsvalid.txt', 10)

    plt_x, err_train, err_valid = net.train(x_train, y_train, x_valid, y_valid,
                                            epochs=epochs, batch_size=100)

    # plot_err(plt_x, err_train, err_valid)


def main(n_layers=1):
    if n_layers == 1:
        net = Net([784, 100, 10], lr=0.1, weight_decay=0, dropout=0.3, momentum=0.5)
        train(net, epochs=250)
    else:
        net = Net([784, 100, 40, 10], lr=0.1, weight_decay=0, dropout=0.3, momentum=0.5)
        train(net, epochs=500)

    # net.save('../result/net.pkl')
    # net = Net.load('../result/net.pkl')

    # w_img = net.visualize_weights((28, 28), 1)
    # img_utils.imshow(w_img)
    # import cv2 as cv
    # cv.imwrite('weights2.jpg', w_img)

    x_test, y_test = read_input('../data/digitstest.txt', 10)
    res = net.predict(x_test)
    err, cerr = net.validate(x_test, y_test)
    print('Test classification error: {:.3f}'.format(ml_utils.classification_error(res, y_test)))
    print(err, cerr)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    n_layers = int(sys.argv[1])

    print(n_layers)

    # main(n_layers)
