import os
import h5py

import numpy as np
from scipy import misc
import keras

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam

import network

import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'


def train():
    #model = keras.models.load_model('./save_test/model62_50.h5')
    model = network.unet_2d()
    #model.summary()

    #output_file = './data.h5'
    #h5f = h5py.File(args.input_data, 'r')

    f = h5py.File('train_data.h5')
    X=f['data']
    f = h5py.File('train_label.h5')
    y = f['data']

    n_epoch = args.n_epoch

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    for epoch in range(0, n_epoch,1):
        model.fit(X, y, batch_size=2, nb_epoch=1, shuffle='batch')
        if args.save:
            print("Saving model ", epoch + 1)
            model.save(os.path.join(args.save, 'test_1_%d.h5' %(epoch+1)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--save',
                        default='./save_test',
                        dest='save',
                        type=str,
                        nargs=1,
                        help="Path to save the checkpoints to")
    parser.add_argument('-D', '--data',
                        default='/home/yx/HSSR_2.0/data/urb_fine_3d_all(300).h5',
                        dest='input_data',
                        type=str,
                        nargs=1,
                        help="Training data directory")
    parser.add_argument('-E', '--epoch',
                        default=50,
                        dest='n_epoch',
                        type=int,
                        nargs=1,
                        help="Training epochs must be a multiple of 5")
    args = parser.parse_args()
    print(args)
    train()
