# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import *
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model

K.set_image_data_format('channels_last')

# weights = [16, 32, 64, 128, 256]
# weights = [8, 16, 32, 64, 128]
weights = [4, 8, 16, 32, 64]

def myConv3D(input_tensor, filters, kernel_size):
    conv1 = Conv3D(filters, kernel_size, padding='same')(input_tensor)
    bn = BatchNormalization()(conv1)
    out = Activation('relu')(bn)
    return out

def get_unet(nClasses, img_depth=19, img_rows=496, img_cols=512):
    inputs = Input((img_depth, img_rows, img_cols, 1))

    # Block 1
    conv1 = myConv3D(inputs, weights[0], (3, 3, 3))
    conv1 = myConv3D(conv1, weights[0], (3, 3, 3))
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), name='block1_pool')(conv1)
    # print('conv1: ', np.shape(conv1))
    # print('pool1: ', np.shape(pool1))

    # Block 2
    conv2 = myConv3D(pool1, weights[1], (3, 3, 3))
    conv2 = myConv3D(conv2, weights[1], (3, 3, 3))
    pool2 = MaxPooling3D(pool_size=(1, 2, 2), name='block2_pool')(conv2)
    # print('conv2: ', np.shape(conv2))
    # print('pool2: ', np.shape(pool2))

    # Block 3
    conv3 = myConv3D(pool2, weights[2], (3, 3, 3))
    conv3 = myConv3D(conv3, weights[2], (3, 3, 3))
    conv3 = myConv3D(conv3, weights[2], (3, 3, 3))
    pool3 = MaxPooling3D(pool_size=(1, 2, 2), name='block3_pool')(conv3)
    # print('conv3: ', np.shape(conv3))
    # print('pool3: ', np.shape(pool3))

    conv4 = myConv3D(pool3, weights[3], (3, 3, 3))
    conv4 = myConv3D(conv4, weights[3], (3, 3, 3))
    conv4 = myConv3D(conv4, weights[3], (3, 3, 3))
    pool4 = MaxPooling3D(pool_size=(1, 2, 2), name='block4_pool')(conv4)
    # print('conv4: ', np.shape(conv4))
    # print('pool4: ', np.shape(pool4))

    conv5 = myConv3D(pool4, weights[4], (3, 3, 3))
    conv5 = Dropout(0.5)(conv5)
    conv5 = myConv3D(conv5, weights[4], (3, 3, 3))
    conv5 = Dropout(0.5)(conv5)
    # print('conv5: ', np.shape(conv5))

    up6 = UpSampling3D(size=(1, 2, 2), name='up_block4')(conv5)
    up6 = myConv3D(up6, weights[3], (3, 3, 3))
    up6 = concatenate([up6, conv4], axis=4)
    # print('up6: ', np.shape(up6))
    conv6 = myConv3D(up6, weights[3], (3, 3, 3))
    conv6 = myConv3D(conv6, weights[3], (3, 3, 3))
    conv6 = myConv3D(conv6, weights[3], (3, 3, 3))
    # print('conv6: ', np.shape(conv6))

    up7 = UpSampling3D(size=(1, 2, 2), name='up_block3')(conv6)
    up7 = myConv3D(up7, weights[2], (3, 3, 3))
    up7 = concatenate([up7, conv3], axis=4)
    # print('up7: ', np.shape(up7))
    conv7 = myConv3D(up7, weights[2], (3, 3, 3))
    conv7 = myConv3D(conv7, weights[2], (3, 3, 3))
    conv7 = myConv3D(conv7, weights[2], (3, 3, 3))
    # print('conv7: ', np.shape(conv7))

    up8 = UpSampling3D(size=(1, 2, 2), name='up_block2')(conv7)
    up8 = myConv3D(up8, weights[1], (3, 3, 3))
    up8 = concatenate([up8, conv2], axis=4)
    # print('up8: ', np.shape(up8))
    conv8 = myConv3D(up8, weights[1], (3, 3, 3))
    conv8 = myConv3D(conv8, weights[1], (3, 3, 3))
    # print('conv8: ', np.shape(conv8))

    up9 = UpSampling3D(size=(1, 2, 2), name='up_block1')(conv8)
    up9 = myConv3D(up9, weights[0], (3, 3, 3))
    up9 = concatenate([up9, conv1], axis=4)
    # print('up9: ', np.shape(up9))
    conv9 = myConv3D(up9, weights[0], (3, 3, 3))
    # print('conv9: ', np.shape(conv9))
    conv9 = myConv3D(conv9, weights[0], (3, 3, 3))
    print('conv9: ', np.shape(conv9))

    # conv10 = Conv3D(4, (1, 1, 1), name='up_out_conv')(conv9)
    # conv10 = Conv3D(nClasses, (1, 1, 1), activation='softmax', padding='same')(conv9)
    conv10 = Conv3D(nClasses, (1, 1, 1), activation='softmax', name='up_out')(conv9)
    print('conv10: ', np.shape(conv10))

    model = Model(inputs, conv10)

    return model
