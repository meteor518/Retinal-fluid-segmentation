from keras.models import *
from keras.layers import *
import os
import tensorflow as tf

file_path = os.path.dirname(os.path.abspath(__file__))
VGG_Weights_path = file_path + "/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

IMAGE_ORDERING = 'channels_last'
weights = [4, 8, 16, 32, 64]
# weights = [16, 32, 64, 128, 256]

# weights = [32, 64, 128, 256, 512]


def VGGSegnet(n_classes, input_height=496, input_width=512, vgg_level=3):
    img_input = Input(shape=(input_height, input_width, 1))

    # Block 1
    x = Conv2D(weights[0], (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = Conv2D(weights[0], (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='VALID', name='block1_pool')(x)
    # print('block1: ', np.shape(x))
    f1 = x

    # Block 2
    x = Conv2D(weights[1], (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[1], (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='VALID', name='block2_pool')(x)
    f2 = x
    # print('block2: ', np.shape(x))

    # Block 3
    x = Conv2D(weights[2], (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[2], (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[2], (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='VALID', name='block3_pool')(x)
    f3 = x
    # print('block3: ', np.shape(x))

    # Block 4
    x = Conv2D(weights[3], (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[3], (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[3], (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='VALID', name='block4_pool')(x)
    x = Dropout(0.5)(x)
    f4 = x
    print('block4: ', np.shape(x))

    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    # # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    # # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), padding='VALID', name='block5_pool')(x)
    # x = Dropout(0.5)(x)
    # f5 = x
    # print('block5: ', np.shape(x))

    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dense(1000, activation='softmax', name='predictions')(x)
    # print('dense: ', np.shape(x))
    #
    # vgg = Model(img_input, x)
    # vgg.load_weights(VGG_Weights_path)

    levels = [f1, f2, f3, f4]

    o = levels[vgg_level]

    # o = unpool_with_argmax(o, arg5, name='unpool5')
    # o = (Conv2D(512, (3, 3), activation='relu', padding='same', name='up_block5_conv1', data_format=IMAGE_ORDERING))(o)
    # o = (Conv2D(512, (3, 3), activation='relu', padding='same', name='up_block5_conv2', data_format=IMAGE_ORDERING))(o)
    # o = (Conv2D(512, (3, 3), activation='relu', padding='same', name='up_block5_conv3', data_format=IMAGE_ORDERING))(o)
    # o = Dropout(0.5)(o)
    # print('o2: ', np.shape(o))

    o = UpSampling2D((2, 2), name='unpool4', data_format=IMAGE_ORDERING)(o)
    o = (Conv2D(weights[3], (3, 3), activation='relu', padding='same', name='up_block4_conv1',
                data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(weights[3], (3, 3), activation='relu', padding='same', name='up_block4_conv2',
                data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(weights[3], (3, 3), activation='relu', padding='same', name='up_block4_conv3',
                data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    # print('o3: ', np.shape(o))

    o = UpSampling2D((2, 2), name='unpool3', data_format=IMAGE_ORDERING)(o)
    o = (Conv2D(weights[2], (3, 3), activation='relu', padding='same', name='up_block3_conv1',
                data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(weights[2], (3, 3), activation='relu', padding='same', name='up_block3_conv2',
                data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(weights[2], (3, 3), activation='relu', padding='same', name='up_block3_conv3',
                data_format=IMAGE_ORDERING))(o)
    # print('o4: ', np.shape(o))

    o = UpSampling2D((2, 2), name='unpool2', data_format=IMAGE_ORDERING)(o)
    o = (Conv2D(weights[1], (3, 3), activation='relu', padding='same', name='up_block2_conv1',
                data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(weights[1], (3, 3), activation='relu', padding='same', name='up_block2_conv2',
                data_format=IMAGE_ORDERING))(o)

    o = UpSampling2D((2, 2), name='unpool1', data_format=IMAGE_ORDERING)(o)
    o = (Conv2D(weights[0], (3, 3), activation='relu', padding='same', name='up_block1_conv1',
                data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(weights[0], (3, 3), activation='relu', padding='same', name='up_block1_conv2',
                data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(n_classes, (1, 1), activation='softmax', padding='same', name='softmax', data_format=IMAGE_ORDERING))(o)

    # o_shape = Model(img_input, o).output_shape
    # outputHeight = o_shape[1]
    # outputWidth = o_shape[2]
    # print('o5: ', np.shape(o))

    # o = (Reshape((-1, outputHeight * outputWidth)))(o)
    # o = (Permute((2, 1)))(o)
    # o = (Activation('softmax'))(o)
    # print('o: ', np.shape(o))
    model = Model(img_input, o)
    # model.outputWidth = outputWidth
    # model.outputHeight = outputHeight

    return model


if __name__ == '__main__':
    m = VGGSegnet(101)
    from keras.utils import plot_model

    plot_model(m, show_shapes=True, to_file='model.png')
