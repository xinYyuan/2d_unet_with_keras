from os import listdir
from os.path import isfile, join
import argparse
import h5py
import math

import numpy as np
from scipy import misc
from sklearn.metrics import  mean_squared_error
import network

import matplotlib.pyplot as plt
#import evaluate
import scipy.io as sio
from keras import backend as K

input_size = 33
label_size = 21
pad = (33 - 21) // 2


class SRCNN(object):
    def __init__(self, weight):
        self.model = network.unet_2d()#(None, None,None,1))
        #self.model = network2d.srcnn2((None, None, 3))
        #self.model.summary()
        #self.model = network2d.srcnn((None, None, 1))
        f = h5py.File(weight, mode='r')
        self.model.load_weights_from_hdf5_group(f['model_weights'])

    def predict(self, data, **kwargs):
        use_3d_input = kwargs.pop('use_3d_input', True)
        if use_3d_input :
            #if data.ndim != 3:
                #raise ValueError('the dimension of data must be 3 !!')
            #channels = data.shape[0]
            #im_out = [self.model.predict(data[i,None, :, :, :, :]) for i in range(channels)]
            im_out = [self.model.predict(data)]
            #get_3rd_layer_output = K.function([self.model.layers[0].input],
            #                                  [self.model.layers[0].output])
            #im_out = get_3rd_layer_output([data])
        else:
            im_out = [self.model.predict(data)]
            if data.ndim != 2:
                raise ValueError('the dimension of data must be 2 !!')
            im_out = self.model.predict(data[None, :, :, None])
        return np.asarray(im_out)

def show_picture(data):
    plt.imshow(data,plt.cm.gray)
    plt.show()

def getto1(img):
    mean = np.mean(img)
    img = img - mean
    min = np.min(img)
    max = np.max(img)
    img = (img - min) / (max - min)
    return img

def predict():



    #X = np.load('test/trainImages.npy').astype(np.float64)

    #f=h5py.File('train_data_withoutseg.h5')

    f = h5py.File('val_data_withoutseg.h5')
    input=f['data']
    input=input[369,:,:,:]
    input=np.reshape(input,[1,512,512,1])
    '''

    input = np.load('validation/images_00201_0086.npy').astype(np.float64)
    print input.shape
    input=np.swapaxes(input,0,1)
    input = np.swapaxes(input, 2, 1)
    input=np.reshape(input[:,:,0],[1,512,512,1])
    print  input.shape
    input=getto1(input)
    '''
    srcnn = SRCNN(option.model)

    output = srcnn.predict(input)
    print np.unique(output)

    print output.shape

    f = h5py.File('val_label_withoutseg.h5')
    label = f['data']
    #print label.shape
    #label=np.load('validation/masks_00002_0146.npy').astype(np.float64)
    show_picture(input[0,:, :, 0 ])
    show_picture(label[369,:,:,0])
    #show_picture(label[0, :, :])
    show_picture(output[0,0, :, :, 0])


    print '123'




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-M', '--model',

                        default='save_test/test_2_15.h5',

                        dest='model',
                        type=str,
                        nargs=1,
                        help="The model to be used for prediction")
    parser.add_argument('-I', '--input-file',
                        default='./dataset/Test/Set5/baby_GT.bmp',
                        dest='input',
                        type=str,
                        nargs=1,
                        help="Input image file path")
    parser.add_argument('-O', '--output-file',
                        default='./dataset/Test/Set5/baby_SRCNN.bmp',
                        dest='output',
                        type=str,
                        nargs=1,
                        help="Output image file path")
    parser.add_argument('-B', '--baseline',
                        default='./dataset/Test/Set5/baby_bicubic.bmp',
                        dest='baseline',
                        type=str,
                        nargs=1,
                        help="Baseline bicubic interpolated image file path")
    parser.add_argument('-S', '--scale-factor',
                        default=2.0,
                        dest='scale',
                        type=float,
                        nargs=1,
                        help="Scale factor")
    option = parser.parse_args()


    predict()