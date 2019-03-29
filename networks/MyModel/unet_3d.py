# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import *
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model

K.set_image_data_format('channels_last')

# weights = [16, 32, 64, 128, 256]
# weights = [8, 16, 32, 64, 128]
weights = [4, 8, 16, 32, 64]


def get_unet(nClasses, img_depth=19, img_rows=496, img_cols=512):
    inputs = Input((img_depth, img_rows, img_cols, 1))

    # Block 1
    conv1 = Conv3D(weights[0], (3, 3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    conv1 = Conv3D(weights[0], (3, 3, 3), activation='relu', padding='same', name='block_conv2')(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), name='block1_pool1')(conv1)
    print('conv1: ', np.shape(conv1))
    print('pool1: ', np.shape(pool1))

    # Block 2
    conv2 = Conv3D(weights[1], (3, 3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    conv2 = Conv3D(weights[1], (3, 3, 3), activation='relu', padding='same', name='block2_conv2')(conv2)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2), name='block2_pool2')(conv2)
    print('conv2: ', np.shape(conv2))
    print('pool2: ', np.shape(pool2))

    # Block 3
    conv3 = Conv3D(weights[2], (3, 3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    conv3 =Conv3D(weights[2], (3, 3, 3), activation='relu', padding='same', name='block3_conv2')(conv3)
    conv3 =Conv3D(weights[2], (3, 3, 3), activation='relu', padding='same', name='block3_conv3')(conv3)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2), name='block3_pool3')(conv3)
    print('conv3: ', np.shape(conv3))
    print('pool3: ', np.shape(pool3))

    conv4 = Conv3D(weights[3], (3, 3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    conv4 = Conv3D(weights[3], (3, 3, 3), activation='relu', padding='same', name='block4_conv2')(conv4)
    conv4 = Conv3D(weights[3], (3, 3, 3), activation='relu', padding='same', name='block4_conv3')(conv4)
    pool4 = MaxPooling3D(pool_size=(1, 2, 2), name='block4_pool4')(conv4)
    print('conv4: ', np.shape(conv4))
    print('pool4: ', np.shape(pool4))

    conv5 = Conv3D(weights[4], (3, 3, 3), activation='relu', padding='same', name='fc_conv1')(pool4)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv3D(weights[4], (3, 3, 3), activation='relu', padding='same', name='fc_conv2')(conv5)
    conv5 = Dropout(0.5)(conv5)
    print('conv5: ', np.shape(conv5))

    up6 = Conv3D(weights[3], (3, 3, 3), activation='relu', padding='same', name='up_block4_upconv')(
        UpSampling3D(size=(1, 2, 2), name='up_block4')(conv5))
    up6 = concatenate([up6, conv4], axis=4)
    print('up6: ', np.shape(up6))
    conv6 = Conv3D(weights[3], (3, 3, 3), activation='relu', padding='same', name='up_block4_conv1')(up6)
    conv6 = Conv3D(weights[3], (3, 3, 3), activation='relu', padding='same', name='up_block4_conv2')(conv6)
    conv6 = Conv3D(weights[3], (3, 3, 3), activation='relu', padding='same', name='up_block4_conv3')(conv6)
    print('conv6: ', np.shape(conv6))

    up7 = Conv3D(weights[2], (3, 3, 3), activation='relu', padding='same', name='up_block3_upconv')(
        UpSampling3D(size=(1, 2, 2), name='up_block3')(conv6))
    up7 = concatenate([up7, conv3], axis=4)
    print('up7: ', np.shape(up7))
    conv7 = Conv3D(weights[2], (3, 3, 3), activation='relu', padding='same', name='up_block3_conv1')(up7)
    conv7 = Conv3D(weights[2], (3, 3, 3), activation='relu', padding='same', name='up_block3_conv2')(conv7)
    conv7 = Conv3D(weights[2], (3, 3, 3), activation='relu', padding='same', name='up_block3_conv3')(conv7)
    print('conv7: ', np.shape(conv7))

    up8 = Conv3D(weights[1], (3, 3, 3), activation='relu', padding='same', name='up_block2_upconv')(
        UpSampling3D(size=(1, 2, 2), name='up_block2')(conv7))
    up8 = concatenate([up8, conv2], axis=4)
    print('up8: ', np.shape(up8))
    conv8 = Conv3D(weights[1], (3, 3, 3), activation='relu', padding='same', name='up_block2_conv1')(up8)
    conv8 = Conv3D(weights[1], (3, 3, 3), activation='relu', padding='same', name='up_block2_conv2')(conv8)
    print('conv8: ', np.shape(conv8))

    up9 = Conv3D(weights[0], (3, 3, 3), activation='relu', padding='same', name='up_block1_upconv')(
        UpSampling3D(size=(1, 2, 2), name='up_block1')(conv8))
    up9 = concatenate([up9, conv1], axis=4)
    print('up9: ', np.shape(up9))
    conv9 = Conv3D(weights[0], (3, 3, 3), activation='relu', padding='same', name='up_block1_conv1')(up9)
    print('conv9: ', np.shape(conv9))
    conv9 = Conv3D(weights[0], (3, 3, 3), activation='relu', padding='same', name='up_block1_conv2')(conv9)
    print('conv9: ', np.shape(conv9))

    # conv10 = Conv3D(4, (1, 1, 1), activation='relu', padding='same', name='up_out_conv')(conv9)
    # conv10 = Conv3D(nClasses, (1, 1, 1), activation='softmax', padding='same')(conv9)
    conv10 = Conv3D(nClasses, (1, 1, 1), activation='softmax', padding='same', name='up_out')(conv9)
    print('conv10: ', np.shape(conv10))

    model = Model(inputs, conv10)

    return model
