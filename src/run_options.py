import logging
import numpy as np

from src import main
from src.neural_network import Net


_lr = [0.1]
_momentum = [0.5]
_dropout = [0.3]
_hidden_units = [100]
_hidden_units_2 = [40]
_w_decay = [0, 0.005]
_batch = [100]
_epoch = [500]


def each_option():
    opt = dict()
    for lr in _lr:
        opt['lr'] = lr
        for momentum in _momentum:
            opt['momentum'] = momentum
            for dropout in _dropout:
                opt['dropout'] = dropout
                for i, n_hidden_units in enumerate(_hidden_units):
                    opt['n_hidden_units'] = n_hidden_units
                    opt['n_hidden_units_2'] = _hidden_units_2[i]
                    for w_decay in _w_decay:
                        opt['w_decay'] = w_decay
                        for batch in _batch:
                            opt['batch'] = batch
                            for n_epoch in _epoch:
                                opt['epochs'] = n_epoch
                                yield opt


def bulk_run():
    # np.random.seed(10)

    x_train, y_train = main.read_input('../data/digitstrain.txt', 10)
    x_valid, y_valid = main.read_input('../data/digitsvalid.txt', 10)

    # x_train = x_train[::2]
    # y_train = y_train[::2]
    # x_valid = x_valid[::2]
    # y_valid = y_valid[::2]

    with open('../result/result-2layer.log', 'a') as f:
        for opt in each_option():
            print('\nStarting options: ', opt)

            # net_arch = [784, opt['n_hidden_units'], 10]

            net_arch = [784, opt['n_hidden_units'], opt['n_hidden_units_2'], 10]

            net = Net(net_arch,
                      lr=opt['lr'],
                      weight_decay=opt['w_decay'],
                      dropout=opt['dropout'],
                      momentum=opt['momentum'])

            plt_x, err_train, err_valid = net.train(x_train, y_train,
                                                    x_valid, y_valid,
                                                    epochs=opt['epochs'], batch_size=opt['batch'])

            print(opt, file=f)
            print(plt_x, file=f)
            print([float('{:.4f}'.format(e)) for e in err_train[0]], file=f)
            print([float('{:.4f}'.format(e)) for e in err_valid[0]], file=f)
            print([float('{:.4f}'.format(e)) for e in err_train[1]], file=f)
            print([float('{:.4f}'.format(e)) for e in err_valid[1]], file=f)
            print('Best val cross-entropy error: {}'.format(min(err_valid[0])), file=f)
            print('Best val classification error: {}'.format(min(err_valid[1])), file=f)
            print('Best cross-entropy error at epoch: {}'.format(plt_x[np.argmin(err_valid[0])]),
                  file=f)
            print('Best classification error at epoch: {}'.format(plt_x[np.argmin(err_valid[1])]),
                  file=f)
            print('----------', file=f)

            main.plot_err(plt_x, err_train, err_valid)


def plot_result():
    plt_x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000]

    opt = {'momentum': 0.5, 'epochs': 1000, 'lr': 0.1, 'dropout': 0, 'w_decay': 0, 'n_hidden_units': 100, 'batch': 100}
    err_train = [0.2993, 0.1304, 0.0766, 0.0502, 0.0351, 0.0262, 0.0206, 0.0168, 0.0142, 0.0123, 0.0108, 0.0096, 0.0086, 0.0078, 0.0072, 0.0066, 0.0061, 0.0057, 0.0053, 0.005, 0.0047, 0.0045, 0.0042, 0.004, 0.0038, 0.0037, 0.0035, 0.0033, 0.0032, 0.0031, 0.003, 0.0029, 0.0028, 0.0027, 0.0026, 0.0025, 0.0024, 0.0023, 0.0023, 0.0022, 0.0021, 0.0021, 0.002, 0.002, 0.0019, 0.0019, 0.0018, 0.0018, 0.0017, 0.0017, 0.0017, 0.0016, 0.0016, 0.0016, 0.0015, 0.0015, 0.0015, 0.0014, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008]
    err_valid = [0.4554, 0.3409, 0.317, 0.3092, 0.3078, 0.3086, 0.3101, 0.3119, 0.3137, 0.3154, 0.3172, 0.3188, 0.3204, 0.3219, 0.3233, 0.3246, 0.3259, 0.3272, 0.3284, 0.3295, 0.3306, 0.3317, 0.3327, 0.3337, 0.3346, 0.3355, 0.3364, 0.3373, 0.3381, 0.3389, 0.3397, 0.3405, 0.3412, 0.3419, 0.3426, 0.3433, 0.344, 0.3446, 0.3453, 0.3459, 0.3465, 0.3471, 0.3477, 0.3483, 0.3488, 0.3494, 0.3499, 0.3504, 0.351, 0.3515, 0.352, 0.3525, 0.3529, 0.3534, 0.3539, 0.3543, 0.3548, 0.3552, 0.3557, 0.3561, 0.3565, 0.3569, 0.3574, 0.3578, 0.3582, 0.3586, 0.3589, 0.3593, 0.3597, 0.3601, 0.3605, 0.3608, 0.3612, 0.3615, 0.3619, 0.3622, 0.3626, 0.3629, 0.3632, 0.3636, 0.3639, 0.3642, 0.3645, 0.3649, 0.3652, 0.3655, 0.3658, 0.3661, 0.3664, 0.3667, 0.367, 0.3673, 0.3676, 0.3678, 0.3681, 0.3684, 0.3687, 0.3689, 0.3692, 0.3695]
    cerr_train = [9.6333, 3.3667, 1.3667, 0.6333, 0.3, 0.0667, 0.0667, 0.0333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cerr_valid = [12.6, 10.2, 9.8, 9.6, 9.1, 8.9, 8.8, 8.6, 8.4, 8.5, 8.2, 8.3, 8.2, 8.2, 8.2, 8.1, 8.1, 8.0, 8.0, 8.0, 8.0, 8.0, 8.1, 8.1, 8.0, 8.0, 8.1, 8.1, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2]

    main.plot_err(plt_x, (err_train, cerr_train), (err_valid, cerr_valid))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    bulk_run()

    # plot_result()
