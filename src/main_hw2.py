import numpy as np
import logging
import matplotlib.pyplot as plt

from cv_utils import img_utils
from src.dataset import MNIST
from src.nn import FullyConnected, Sigmoid, SoftMax, Sequential, Solver, Dropout, rbm
from src import ml_utils
from src.nn.solver import AutoEncoderSolver


def gen_rbm_image(net, dataset):
    inp_imgs = list()
    gen_imgs = list()
    gen_imgs2 = list()

    for i in range(0, 1000, 100):
        x = dataset.x_valid[i:i+1, :]

        # x = np.random.randint(0, 2, (1, 28*28))
        img = np.reshape(x, (28, 28))
        inp_imgs.append(img)

        x_gen, h_gen = net.gibbs_sampling(x, steps=1000)
        img_gen = np.reshape(x_gen, (28, 28))
        gen_imgs.append(img_gen)

        x_gen = net.backward(h_gen)
        img_gen = np.reshape(x_gen, (28, 28))
        gen_imgs2.append(img_gen.round())

    img_utils.imshow(*(inp_imgs + gen_imgs + gen_imgs2), cmap='gray')


def plot_err(plt_x, err_train, err_valid):
    plt.plot(plt_x, err_train, 'bs', label='Training error')
    plt.plot(plt_x, err_valid, 'g^', label='Validation error')
    plt.ylabel('Cross-entropy error')
    plt.xlabel('Epochs')
    # plt.ylim((0, 1))
    plt.legend()
    plt.show()
    plt.close('all')


def rbm_main():
    dataset = MNIST('../data/digitstrain.txt', '../data/digitstest.txt', '../data/digitsvalid.txt')
    # dataset.x_train = np.vstack((dataset.x_train, dataset.x_test))
    dataset.x_test = []
    dataset.binarize()
    dataset.shuffle()

    net = rbm.RBM('rbm', 28*28, 100)

    # plt_x, t_errs, v_errs = list(), list(), list()
    for epoch in rbm.solver(net, dataset, epochs=100, lr=0.01, bsize=10):
        if epoch % 5 == 0:
            print('epoch', epoch)
            _, h_gen = net.gibbs_sampling(dataset.x_train, 1)
            x_gen = net.backward(h_gen)
            err = np.sum(((dataset.x_train - x_gen) ** 2) / dataset.x_train.shape[0])
            logging.info('Training Loss : {}'.format(err))

            err = ml_utils.cross_entropy_reconstruction_error(dataset.x_train, x_gen)
            logging.info('Training Cross-entropy error: {}'.format(err))

            # x_gen, _ = net.gibbs_sampling(dataset.x_valid, 1)
            _, h_gen = net.gibbs_sampling(dataset.x_valid, 1)
            x_gen = net.backward(h_gen)
            valid_err = np.sum(((dataset.x_valid - x_gen) ** 2) / dataset.x_valid.shape[0])
            logging.info('Validation Loss : {}'.format(valid_err))

            err = ml_utils.cross_entropy_reconstruction_error(dataset.x_valid, x_gen)
            logging.info('Validation Cross-entropy error: {}'.format(err))

            #     plt_x.append(epoch)
            #     t_errs.append(err)
            #     v_errs.append(valid_err)

        if epoch % 20 == 0:
            gen_rbm_image(net, dataset)
            w_img = net.fc.visualize_weight()
            img_utils.imshow(w_img)

    # net.save('../result2/rbm.pkl')

    # plot_err(plt_x, t_errs, v_errs)

    # net = Net.load('../result2/rbm.pkl')


def classify():
    dataset = MNIST('../data/digitstrain.txt', '../data/digitstest.txt', '../data/digitsvalid.txt')
    dataset.normalize()

    modules = [
        FullyConnected('fc1', 28*28, 100),
        Sigmoid('sigm1'),
        FullyConnected('fc2', 100, 10),
        SoftMax('smax')
    ]
    net = Sequential('net', modules)
    solver = Solver(net, 0.1, epochs=200, batch_size=100)
    solver.train(dataset)

    err, cerr = solver.validate(dataset.x_test, dataset.y_test)
    print(err, cerr)


def gen_ae_imgs(net, dataset):
    inp_imgs = list()
    gen_imgs = list()
    for i in range(0, 1000, 100):
        x = dataset.x_valid[i:i+1, :]

        img = np.reshape(x, (28, 28))
        inp_imgs.append(img)

        x_gen = net.forward(x)
        img_gen = np.reshape(x_gen, (28, 28))
        gen_imgs.append(img_gen)

    img_utils.imshow(*(inp_imgs + gen_imgs), cmap='gray')


def auto_encoder():
    dataset = MNIST('../data/digitstrain.txt', '../data/digitstest.txt', '../data/digitsvalid.txt')
    dataset.y_train = dataset.x_train
    dataset.y_valid = dataset.x_valid
    dataset.y_test = dataset.x_test

    dataset.binarize()
    dataset.shuffle()

    def train_ae(net):
        solver = AutoEncoderSolver(net, 0.01, epochs=100, batch_size=100, test_interval=5)
        solver.train(dataset)

        # gen_ae_imgs(net, dataset)

        w_img = net.find('fc1').visualize_weight()
        img_utils.imshow(w_img)

    fc1 = FullyConnected('fc1', 28*28, 100)
    fc2 = FullyConnected('fc2', 100, 28*28, weights=[fc1.w.T])
    modules = [
        fc1, Sigmoid('sigm1'),
        fc2, Sigmoid('sigm2'),
    ]

    net = Sequential('autoencoder', modules)
    train_ae(net)
    net.save('../result2/autoencoder.pkl')

    net.modules.insert(0, Dropout('drop1', percent=0.5))
    train_ae(net)
    net.save('../result2/denoising-autoencoder.pkl')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # rbm_main()

    # classify()

    auto_encoder()

    exit()
