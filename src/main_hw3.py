import logging
import pickle as pkl
import cv2 as cv
import os.path as osp
import cv_utils
import math
import numpy as np

from cv_utils import img_utils
from src.dataset import MNIST
from src.nn import dbm, Solver, GenSolver
from src.nn.rbm import RBM


def convert_mat2collage(im_x, h, w, val_range=(0, 1)):
    img_mat = np.reshape(im_x, (im_x.shape[0], h, w))
    imgs = [img_utils.normalize_img(img_mat[i, :, :], val_range)
            for i in range(im_x.shape[0])]
    size = int(math.sqrt(len(imgs)))
    coll = img_utils.collage(imgs, (size, size))
    return coll


def load_model(name, save_dir, epoch):
    fname = osp.join(save_dir, '{}-ep{:03d}.pkl'.format(name, epoch))
    with open(fname, 'rb') as f:
        return pkl.load(f)


def generate_snapshot(model):
    x = dataset.x_train[:100]
    # x = np.random.randint(0, 2, (100, 28*28))
    x_gen, x_prob = model.reconstruct(x, steps=1000)

    inp = convert_mat2collage(x, 28, 28)
    gen = convert_mat2collage(x_gen, 28, 28)

    x_prob[x_prob < 0.4] = 0
    x_prob[x_prob >= 0.4] = 1
    gen_pb = convert_mat2collage(x_prob.round(), 28, 28)

    # weights visualization
    fc1 = model.fc if isinstance(model, RBM) else model.layers[1].fc
    w_img = fc1.visualize_weight()

    save_im = img_utils.collage([inp, gen, gen_pb, w_img], (2, 2),
                                bg=cv_utils.COL_WHITE)
    return save_im


def train_pipeline():
    def view_samples(solver, epoch):
        pcd = convert_mat2collage(solver.net.pcd, 28, 28)
        cv.imwrite(osp.join(solver.save_dir, 'pcd-{}-ep{}.jpg'
                            .format(solver.net.name, epoch)), pcd)

    def gen(solver, epoch):
        save_im = generate_snapshot(solver.net)
        cv.imwrite(osp.join(solver.save_dir, 'shot-{}-ep{}.jpg'
                            .format(solver.net.name, epoch)), save_im)

    def decay_lr(solver, epoch):
        solver.lr /= 1.5

    freq = 5
    return [
        (view_samples, freq),
        (gen, freq),
        (decay_lr, 50)
    ]


def create_rbm(name, n_chain=100):
    model = RBM(name, 28*28, 100, n_chain)
    return model


def create_dbm(name, n_chain=100):
    h1_size, h2_size = 200, 200
    h2 = dbm.DBMLayer('h2', h1_size, h2_size, n_chain)
    h1 = dbm.DBMLayer('h1', 28*28, h1_size, n_chain)
    h0 = dbm.DBMLayer0('h0', 28*28, n_chain)
    model = dbm.DBM(name, [h0, h1, h2])
    return model


def train(model, epochs):
    lr = 0.01
    batch_size = 5
    momentum = 0.5

    solver = GenSolver(model, lr, epochs, batch_size, momentum, test_interval=5)
    solver.add_save_pipe(save_dir, save_freq=50)
    solver.pipeline.extend(train_pipeline())
    solver.train(dataset)


def main():
    name = 'dbm'
    n_chain = 100
    epochs = (1, 30)

    # reload saved model
    # model = Solver.load_model(name, 100, save_dir)

    model = create_dbm(name, n_chain)
    # model = create_rbm(name, n_chain)

    train(model, epochs)

    save_im = generate_snapshot(model)
    # cv.imwrite(osp.join(save_dir, 'shot-{}.jpg'.format(model.name)), save_im)
    img_utils.imshow(save_im)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    np.random.seed(32)

    datadir = 'data/'
    save_dir = 'result3/'

    dataset = MNIST(osp.join(datadir, 'digitstrain.txt'),
                    osp.join(datadir, 'digitstest.txt'),
                    osp.join(datadir, 'digitsvalid.txt'))

    # dataset.x_train = np.delete(dataset.x_train, range(600), 0)
    # dataset.y_train = np.delete(dataset.y_train, range(600), 0)
    #
    # dataset.x_valid = np.delete(dataset.x_valid, range(200), 0)
    # dataset.y_valid = np.delete(dataset.y_valid, range(200), 0)

    dataset.x_train = np.vstack((dataset.x_train, dataset.x_test))
    dataset.y_train = np.vstack((dataset.y_train, dataset.y_test))

    dataset.binarize()
    dataset.shuffle()

    main()
