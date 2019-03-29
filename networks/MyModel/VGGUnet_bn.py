from keras.models import *
from keras.layers import *

import os

file_path = os.path.dirname(os.path.abspath(__file__))

VGG_Weights_path = file_path + "/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

IMAGE_ORDERING = 'channels_last'
# weights = [8, 16, 32, 64, 128]
# weights = [32, 64, 128, 256, 512]
# weights = [16, 32, 64, 128, 256]
weights = [4, 8, 16, 32, 64]

def myConv2D(input_tensor, filters, kernel_size):
    conv1 = Conv2D(filters, kernel_size, padding='same', data_format=IMAGE_ORDERING)(input_tensor)
    bn = BatchNormalization()(conv1)
    out = Activation('relu')(bn)
    return out

def VGGUnet(n_classes, input_height=496, input_width=512, vgg_level=4):
    # assert input_height % 32 == 0
    # assert input_width % 32 == 0
    img_input = Input(shape=(input_height, input_width, 1))
    # print(img_input.shape)

    # Block 1
    x = myConv2D(img_input, weights[0], (3, 3))
    x = myConv2D(x, weights[0], (3, 3))
    f1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    # print('block1', x.shape)

    # Block 2
    x = myConv2D(x, weights[1], (3, 3))
    x = myConv2D(x, weights[1], (3, 3))
    f2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    # print('block2', x.shape)

    # Block 3
    x = myConv2D(x, weights[2], (3, 3))
    x = myConv2D(x, weights[2], (3, 3))
    x = myConv2D(x, weights[2], (3, 3))
    f3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    # print('block3', x.shape)

    # Block 4
    x = myConv2D(x, weights[3], (3, 3))
    x = myConv2D(x, weights[3], (3, 3))
    x = myConv2D(x, weights[3], (3, 3))
    x = Dropout(0.5)(x)
    f4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)
    print('block4', x.shape)

    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    # x = Dropout(0.5)(x)
    # f5 = x
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)
    # print('block5', x.shape)

    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dense(1000, activation='softmax', name='predictions')(x)
    # print('dense', x.shape)
    #
    # vgg = Model(img_input, x)
    # vgg.load_weights(VGG_Weights_path)

    # print('o1', o.shape)


    x = myConv2D(x, weights[4], (3, 3))
    x = Dropout(0.5)(x)
    x = myConv2D(x, weights[4], (3, 3))
    x = Dropout(0.5)(x)
    f5 = x

    levels = [f1, f2, f3, f4, f5]

    o = levels[vgg_level]
    # print('o2', o.shape)

    o = (UpSampling2D((2, 2), name='up5', data_format=IMAGE_ORDERING))(o)
    o = myConv2D(o, weights[3], (3, 3))
    o = (concatenate([o, f4], axis=3))
    o = myConv2D(o, weights[3], (3, 3))
    o = myConv2D(o, weights[3], (3, 3))
    o = myConv2D(o, weights[3], (3, 3))
    # o = Dropout(0.5)(o)
    # print('o3', o.shape)

    o = (UpSampling2D((2, 2), name='up4', data_format=IMAGE_ORDERING))(o)
    o = myConv2D(o, weights[2], (3, 3))
    o = (concatenate([o, f3], axis=3))
    o = myConv2D(o, weights[2], (3, 3))
    o = myConv2D(o, weights[2], (3, 3))
    o = myConv2D(o, weights[2], (3, 3))
    # o = Dropout(0.5)(o)
    # print('up_block3', o.shape)

    o = (UpSampling2D((2, 2), name='up3', data_format=IMAGE_ORDERING))(o)
    o = myConv2D(o, weights[1], (3, 3))
    o = (concatenate([o, f2], axis=3))
    o = myConv2D(o, weights[1], (3, 3))
    o = myConv2D(o, weights[1], (3, 3))
    # print('up_block2', o.shape)

    o = (UpSampling2D((2, 2), name='up2', data_format=IMAGE_ORDERING))(o)
    o = myConv2D(o, weights[0], (3, 3))
    o = (concatenate([o, f1], axis=3))
    o = myConv2D(o, weights[0], (3, 3))
    o = myConv2D(o, weights[0], (3, 3))
    # print('up_block1', o.shape)

    # o = Conv2D(8, 1, padding='same', activation='relu', name='up_out_conv1', data_format=IMAGE_ORDERING)(o)
    o = Conv2D(n_classes, 1, padding='same', activation='softmax', name='up_out', data_format=IMAGE_ORDERING)(o)
    # o_shape = Model(img_input, o).output_shape
    # outputHeight = o_shape[1]
    # outputWidth = o_shape[2]
    # print('out_o', o_shape, outputHeight, outputWidth)

    # o = (Reshape((n_classes, outputHeight * outputWidth)))(o)
    # o = (Permute((2, 1)))(o)
    # o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    # model.outputWidth = outputWidth
    # model.outputHeight = outputHeight

    return model