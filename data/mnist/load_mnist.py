import sys
import os
import glob
import pickle
import numpy as np
import struct
from PyCmpltrtok.common import *


def _load(xbase_url, xtrain, xtest, xtrain_size, xtest_size):
    """

    """
    sep('Load Cifar10 dataset')

    # decide data dir
    BASE_DIR, FILE_NAME = os.path.split(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, '_data')
    os.makedirs(data_dir, exist_ok=True)

    # download the tar
    for xname, xsize in zip([*xtrain,  *xtest], [*xtrain_size, *xtest_size]):
        tar_path = os.path.join(data_dir, xname)
        if os.path.exists(tar_path):
            xact_size = os.path.getsize(tar_path)
            if xsize == xact_size:
                print(f'{tar_path} already downloaded')
                continue
            elif xsize < xact_size:
                print(f'{tar_path} is bigger than expected. Something should be wrong. Stop.', flush=True, file=sys.stderr)
                sys.exit(1)
        xurl = xbase_url + '/' + xname
        print(f'Downloading ...')
        xcmd = f'wget -O "{tar_path}" -c "{xurl}"'
        print(xcmd)
        os.system(xcmd)
        print('Downloaded.')

    # decompress the tar
    for xname in [*xtrain,  *xtest]:
        tar_path = os.path.join(data_dir, xname)
        tgt_path = glob.glob(data_dir + '/' + xname[:-3])
        n_tgt = len(tgt_path)
        if n_tgt >= 1:
            print(f'{tgt_path} is already there.')
        else:
            print('Decompressing ...')
            xcmd = f'gunzip -f -k "{tar_path}"'
            print(xcmd)
            os.system(xcmd)
            print('Decompressed.')

    # load data
    def load_data(type, xname):
        tgt_path = os.path.join(data_dir, xname[:-3])
        is_label = xname.split('-')[1] == 'labels'
        xdata = []
        with open(tgt_path, 'rb') as f:
            xbytes = f.read(8)
            # https://docs.python.org/3/library/struct.html#struct-format-strings
            mn, num = struct.unpack('>ii', xbytes)
            if is_label:
                for _ in range(num):
                    xlabel = struct.unpack('>B', f.read(1))
                    xdata.append(xlabel)
                xdata = np.array(xdata, dtype=np.uint8)
            else:
                xbytes = f.read(8)
                n_rows, n_cols = struct.unpack('>ii', xbytes)
                for _ in range(num):
                    buff = f.read(n_rows * n_cols)
                    xone = np.frombuffer(buff, dtype=np.uint8).reshape(n_rows, n_cols)
                    xdata.append(xone)
                xdata = np.array(xdata, dtype=np.uint8)
        return xdata

    xdict = {}
    for xname, type in zip([*xtrain,  *xtest], ['x_train', 'y_train', 'x_test', 'y_test']):
        # https://stackoverflow.com/questions/8028708/dynamically-set-local-variable
        xdict[type] = load_data(type, xname)

    return xdict['x_train'], xdict['y_train'], xdict['x_test'], xdict['y_test']


def load(only_meta=False):
    xbase_url = 'http://yann.lecun.com/exdb/mnist'
    xtrain = (
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz'
    )
    xtrain_size = (9912422, 28881)
    xtest = (
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    )
    xtest_size = (1648877, 4542)
    return _load(xbase_url, xtrain, xtest, xtrain_size, xtest_size)


shape_ = (28, 28)  # The shape of very image


if '__main__' == __name__:
    import itertools
    import matplotlib.pyplot as plt

    def _main():
        """
        To test the data loading procedure for MNIST.

        :return: None
        """
        x_train, y_train, x_test, y_test = load()
        sep('Data loaded')
        for type1, type2 in itertools.product(['x', 'y'], ['train', 'test']):
            var_name = f'{type1}_{type2}'
            xx = locals()[var_name]
            print(var_name, type(xx), xx.shape, xx.dtype)

        sep('Check data')
        def check(x_, y_, type2):
            print(f'Checking x_{type2} and y_{type2} ...')
            plt.figure(figsize=[16, 8])
            spn = 0
            spr = 5
            spc = 10

            def check_i(i):
                nonlocal spn
                spn += 1
                plt.subplot(spr, spc, spn)
                plt.axis('off')
                plt.title(f'{i}: {y_[i]}')
                plt.imshow(x_[i].reshape(*shape_))

            half = spr * spc // 2
            for i in range(half):
                check_i(i)

            for i in range(half):
                check_i(-(i + 1))

            print('Check and close the plotting window to continue ...')
            plt.show()

        for type2 in ['train', 'test']:
            check(locals()[f'x_{type2}'], locals()[f'y_{type2}'], type2)
        print('Over! Data loaded and checked!')

    _main()  # Main program entrance
    sep('All over')
