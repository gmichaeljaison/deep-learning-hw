import numpy as np
import logging
import pickle as pkl

from cv_utils import img_utils
from src.neural_network import Net
from src import ml_utils
from matplotlib import pyplot as plt


def read_input(fname, num_class, normalize=True, prob=True):
    data = np.loadtxt(fname, delimiter=',')
    y = data[:, -1]
    x = data[:, :-1]

    if normalize:
        x = ml_utils.normalize_data(x)

    if prob:
        y = ml_utils.lbl2prob(y, num_class)

    return x, y


def disp_image(vec):
    img = np.reshape(vec, (28, 28))
    img_utils.imshow(img)


def train(net):
    x_train, y_train = read_input('../data/digitstrain.txt', 10)
    x_valid, y_valid = read_input('../data/digitsvalid.txt', 10)

    net.train(x_train, y_train, x_valid, y_valid, epochs=1000, batch_size=100, visualize=True)

    plt.close('all')

    with open('../result/net.pkl', 'wb') as f:
        pkl.dump(net, f)


def main():
    # net = Net([784, 100, 10], lr=0.1, weight_decay=0.5)
    # train(net)

    with open('../result/net.pkl', 'rb') as f:
        net = pkl.load(f)

    # w_img = net.visualize_weights((28, 28), 1)
    # img_utils.imshow(w_img)

    x_test, y_test = read_input('../data/digitstest.txt', 10)
    res = net.predict(x_test)
    print('Test accuracy: {:.3f}'.format(ml_utils.classification_error(res, y_test)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
