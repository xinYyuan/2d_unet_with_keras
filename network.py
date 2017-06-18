from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Convolution2D ,Convolution3D,MaxPooling3D,Activation,PReLU,LeakyReLU,MaxPooling2D,merge,UpSampling2D,BatchNormalization,Dropout
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import numpy as np
from keras.optimizers import Adam,SGD
import argparse

img_rows = 512
img_cols = 512

#smooth = 100.
smooth = 0
from keras import backend as K
import sys



def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)




def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    smooth = 0.
    intersection = K.sum(y_true * y_pred)

    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def get_options():
    parser = argparse.ArgumentParser(description='UNET for Lung Nodule Detection')





    parser.add_argument('-lr', action="store", default=0.0001, dest="lr", type=float)

    parser.add_argument('-filter_width', action="store", default=3, dest="filter_width", type=int)
    parser.add_argument('-stride', action="store", default=3, dest="stride", type=int)

    opts = parser.parse_args(sys.argv[1:])

    return opts


def unet_2d():
    print '1'

    options = get_options()
    inputs = Input(( 512, 512,1))
    conv1 = Convolution2D(32, options.filter_width, options.stride, activation='elu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_1')(
        conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool_1')(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Convolution2D(64, options.filter_width, options.stride, activation='elu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_2')(
        conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool_2')(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Convolution2D(128, options.filter_width, options.stride, activation='elu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_3')(
        conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool_3')(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Convolution2D(256, options.filter_width, options.stride, activation='elu', border_mode='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(256, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_4')(
        conv4)
    conv4 = BatchNormalization()(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2), name='pool_4')(conv4)

    # conv5 = Convolution2D(512, options.filter_width, options.stride, activation='elu',border_mode='same')(pool4)
    # conv5 = Dropout(0.2)(conv5)
    # conv5 = Convolution2D(512, options.filter_width, options.stride, activation='elu',border_mode='same', name='conv_5')(conv5)

    # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    # conv6 = Convolution2D(256, options.filter_width, options.stride, activation='elu',border_mode='same')(up6)
    # conv6 = Dropout(0.2)(conv6)
    # conv6 = Convolution2D(256, options.filter_width, options.stride, activation='elu',border_mode='same', name='conv_6')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=3)

    conv7 = Convolution2D(128, options.filter_width, options.stride, activation='elu', border_mode='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(128, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_7')(
        conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Convolution2D(64, options.filter_width, options.stride, activation='elu', border_mode='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(64, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_8')(
        conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Convolution2D(32, options.filter_width, options.stride, activation='elu', border_mode='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(32, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_9')(
        conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', name='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    #model.summary()
    model.compile(optimizer=Adam(lr=options.lr, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])

    return model




def unet_2d2():
    print '1'

    options = get_options()
    inputs = Input(( 512, 512,1))
    conv1 = Convolution2D(32, options.filter_width, options.stride, activation='elu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_1')(
        conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool_1')(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Convolution2D(64, options.filter_width, options.stride, activation='elu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_2')(
        conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool_2')(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Convolution2D(128, options.filter_width, options.stride, activation='elu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_3')(
        conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool_3')(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Convolution2D(256, options.filter_width, options.stride, activation='elu', border_mode='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(256, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_4')(
        conv4)
    conv4 = BatchNormalization()(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2), name='pool_4')(conv4)

    # conv5 = Convolution2D(512, options.filter_width, options.stride, activation='elu',border_mode='same')(pool4)
    # conv5 = Dropout(0.2)(conv5)
    # conv5 = Convolution2D(512, options.filter_width, options.stride, activation='elu',border_mode='same', name='conv_5')(conv5)

    # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    # conv6 = Convolution2D(256, options.filter_width, options.stride, activation='elu',border_mode='same')(up6)
    # conv6 = Dropout(0.2)(conv6)
    # conv6 = Convolution2D(256, options.filter_width, options.stride, activation='elu',border_mode='same', name='conv_6')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=3)

    conv7 = Convolution2D(128, options.filter_width, options.stride, activation='elu', border_mode='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(128, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_7')(
        conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Convolution2D(64, options.filter_width, options.stride, activation='elu', border_mode='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(64, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_8')(
        conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Convolution2D(32, options.filter_width, options.stride, activation='elu', border_mode='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(32, options.filter_width, options.stride, activation='elu', border_mode='same', name='conv_9')(
        conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', name='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    #model.summary()
    model.compile(optimizer=Adam(lr=0.001, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])

    return model