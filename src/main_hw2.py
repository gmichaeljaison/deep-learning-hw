import numpy as np
import logging
import matplotlib.pyplot as plt

from cv_utils import img_utils
from src.dataset import MNIST
from src import ml_utils
from src.nn import FullyConnected, Sigmoid, SoftMax, Sequential, Solver, AutoEncoderSolver, \
    Dropout, rbm, Module


def gen_rbm_images(dataset, net_fpath):
    net = Module.load(net_fpath)
    gen_rbm_image(net, dataset, 100)


def gen_rbm_image(net, dataset, n):
    inp_imgs = list()
    gen_imgs = list()

    for i in range(0, 3000, 3000 // n):
        x = dataset.x_test[i:i+1, :]

        # x = np.random.randint(0, 2, (1, 28*28))
        img = np.reshape(x, (28, 28))
        inp_imgs.append(img)

        _, h_gen = net.gibbs_sampling(x, steps=1000)
        x_gen = net.backward(h_gen)
        img_gen = np.reshape(x_gen, (28, 28))
        img_gen[img_gen > 0.3] = 1
        gen_imgs.append(img_gen.round())

    img_utils.imshow(*inp_imgs, cmap='gray')
    img_utils.imshow(*gen_imgs, cmap='gray')

    # img_utils.imshow(*(inp_imgs + gen_imgs), cmap='gray')


def plot_err(plt_x, err_train, err_valid):
    plt.plot(plt_x, err_train, 'bs', label='Training error')
    plt.plot(plt_x, err_valid, 'g^', label='Validation error')
    plt.ylabel('Cross-entropy error')
    plt.xlabel('Epochs')
    plt.ylim((min(0.5, min(err_train), min(err_valid)), max(0.22, max(err_train), max(err_valid))))
    plt.legend()
    plt.show()
    plt.close('all')


def rbm_main(dataset):
    net = rbm.RBM('rbm', 28*28, 500)
    net.train_steps = 20

    plt_x, t_errs, v_errs = list(), list(), list()
    for epoch in rbm.solver(net, dataset, epochs=100, lr=0.01, bsize=10):
        if epoch % 5 == 0:
            print('epoch', epoch)
            plt_x.append(epoch)

            _, h_gen = net.gibbs_sampling(dataset.x_train, 1)
            x_gen = net.backward(h_gen)
            err = ml_utils.cross_entropy_reconstruction_error(dataset.x_train, x_gen)
            logging.info('Training Cross-entropy error: {}'.format(err))
            t_errs.append(err)

            _, h_gen = net.gibbs_sampling(dataset.x_valid, 1)
            x_gen = net.backward(h_gen)
            err = ml_utils.cross_entropy_reconstruction_error(dataset.x_valid, x_gen)
            logging.info('Validation Cross-entropy error: {}'.format(err))
            v_errs.append(err)

        # if epoch % 20 == 0:
        #     gen_rbm_image(net, dataset)
        #     w_img = net.fc.visualize_weight()
        #     img_utils.imshow(w_img)

    # net.save('../result2/rbm-steps5.pkl')

    plot_err(plt_x, t_errs, v_errs)
    with open('result2.log', 'a') as f:
        f.write('\n{}\n{}\n{}\n'.format(plt_x, t_errs, v_errs))

    # net = Net.load('../result2/rbm.pkl')


def classify(dataset, net_fname=None):
    weights = None
    if net_fname:
        fnet = Module.load(net_fname)
        fc = fnet.find('fc1')
        weights = [fc.w, fc.bias]

    fc1 = FullyConnected('fc1', 28 * 28, 100)
    fc1.set_weights(weights)
    modules = [
        fc1,
        Sigmoid('sigm1'),
        FullyConnected('fc2', 100, 10),
        SoftMax('smax')
    ]
    net = Sequential('net', modules)
    solver = Solver(net, 0.1, epochs=300, batch_size=50)
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


def auto_encoder(dataset):
    dataset.y_train = dataset.x_train
    dataset.y_valid = dataset.x_valid
    dataset.y_test = dataset.x_test

    def train_ae(net):
        solver = AutoEncoderSolver(net, 0.01, epochs=100, batch_size=50, test_interval=5)
        errs = solver.train(dataset)

        plt_x = [x[0] for x in errs]
        t_err = [x[1] for x in errs]
        v_err = [x[2] for x in errs]

        plot_err(plt_x, t_err, v_err)
        plt.close('all')

        gen_ae_imgs(net, dataset)

        w_img = net.find('fc1').visualize_weight()
        img_utils.imshow(w_img)

    n_h = 100
    fc1 = FullyConnected('fc1', 28*28, n_h)
    fc2 = FullyConnected('fc2', n_h, 28*28, weights=[fc1.w.T])
    modules = [
        fc1, Sigmoid('sigm1'),
        fc2, Sigmoid('sigm2'),
    ]

    net = Sequential('autoencoder', modules)
    train_ae(net)
    # net.save('../result2/autoencoder.pkl')

    # net.modules.insert(0, Dropout('drop1', percent=0.5))
    # train_ae(net)
    # net.save('../result2/denoising-autoencoder.pkl')


def vis_weight(fname):
    net = Module.load(fname)
    w_img = net.find('fc1').visualize_weight()
    img_utils.imshow(w_img)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    dataset = MNIST('../data/digitstrain.txt', '../data/digitstest.txt', '../data/digitsvalid.txt')
    dataset.binarize()
    dataset.shuffle()

    # rbm_main(dataset)
    auto_encoder(dataset)

    # classify(dataset)
    # classify(dataset, '../result2/autoencoder.pkl')
    # classify(dataset, '../result2/denoising-autoencoder.pkl')

    # vis_weight('../result2/denoising-autoencoder.pkl')

    # gen_rbm_images(dataset, '../result2/rbm-steps5.pkl')

    # with open('result2.log') as f:
    #     f.readline()
    #     f.readline()
    #     f.readline()
    #     f.readline()
    #     plt_x = eval(f.readline())
    #     t_err = eval(f.readline())
    #     v_err = eval(f.readline())
    #
    #     print(plt_x)
    #     print(t_err)
    #     print(v_err)
    #
    #     plot_err(plt_x, t_err, v_err)

    plt.close('all')
    exit()
