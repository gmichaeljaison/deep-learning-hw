import logging
import pickle as pkl
import cv2 as cv
import os.path as osp
import cv_utils
import math
import numpy as np

from cv_utils import img_utils
from src.dataset import MNIST
from src.nn import dbm


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


def dbm_train(save_dir, epochs, model=None):
    if model is None:
        n_chains = 100

        h2 = dbm.DBMLayer('h2', 100, 100, n_chains)
        h1 = dbm.DBMLayer('h1', 28*28, 100, n_chains)
        h0 = dbm.DBMLayer0('h0', 28*28, n_chains)
        model = dbm.DBM('dbm-b5-ch100', [h0, h1, h2])

    solver = dbm.DBMSolver(model, lr=0.01, epochs=epochs, batch_size=5,
                           test_interval=5)
    solver.train(dataset, save_dir, save_freq=10)

    return model


def dbm_main():
    save_dir = 'result3/'

    # retrain from snapshot
    model = load_model('dbm-b5-no01', save_dir, 30)
    # model = dbm_train(save_dir, (41, 100), model)

    # model = dbm_train(save_dir, epochs=20)

    # pcd = convert_mat2collage(model.layers[0].h_cap, 28, 28)
    # img_utils.imshow(pcd)

    x = dataset.x_train[100:200]
    # x = np.random.randint(0, 2, (100, 28*28))
    x_gen, x_prob = model.reconstruct(x, 1000)

    inp = convert_mat2collage(x, 28, 28)
    gen = convert_mat2collage(x_gen, 28, 28)
    gen_pb = convert_mat2collage(x_prob.round(), 28, 28)

    # weights visualization
    w_img = model.layers[1].fc.visualize_weight()

    save_im = img_utils.collage([inp, gen, gen_pb, w_img], (2, 2),
                                bg=cv_utils.COL_WHITE)
    # cv.imwrite(osp.join(save_dir, 'shot-{}.jpg'.format(model.name)), save_im)

    img_utils.imshow(save_im)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    datadir = 'data/'

    dataset = MNIST(osp.join(datadir, 'digitstrain.txt'),
                    osp.join(datadir, 'digitstest.txt'),
                    osp.join(datadir, 'digitsvalid.txt'))
    # dataset = MNIST(osp.join(datadir, 'digitstrain.txt'))

    # dataset.x_train = np.delete(dataset.x_train, range(600), 0)
    # dataset.y_train = np.delete(dataset.y_train, range(600), 0)
    #
    # dataset.x_valid = np.delete(dataset.x_valid, range(200), 0)
    # dataset.y_valid = np.delete(dataset.y_valid, range(200), 0)

    dataset.binarize()
    dataset.shuffle()

    # dataset.x_train = np.vstack((dataset.x_train, dataset.x_test))
    # dataset.y_train = np.vstack((dataset.y_train, dataset.y_test))

    dbm_main()
