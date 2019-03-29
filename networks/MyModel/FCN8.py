from keras.models import *
from keras.layers import *
import os

file_path = os.path.dirname(os.path.abspath(__file__))

VGG_Weights_path = file_path + "/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

IMAGE_ORDERING = 'channels_last'
weights = [4, 8, 16, 32, 64]
# weights = [16, 32, 64, 128, 256]
# weights = [32, 64, 128, 256, 512]


# crop o1 wrt o2
def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]

    cx = abs(outputWidth1 - outputWidth2)
    cy = abs(outputHeight2 - outputHeight1)

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D(cropping=((0, 0), (0, cx)), data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0), (0, cx)), data_format=IMAGE_ORDERING)(o2)

    if outputHeight1 > outputHeight2:
        o1 = Cropping2D(cropping=((0, cy), (0, 0)), data_format=IMAGE_ORDERING)(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy), (0, 0)), data_format=IMAGE_ORDERING)(o2)

    return o1, o2


def FCN8(nClasses, input_height=496, input_width=512, vgg_level=3):
    # assert input_height%32 == 0
    # assert input_width%32 == 0

    img_input = Input(shape=(input_height, input_width, 1))

    # Block 1
    x = Conv2D(weights[0], (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = Conv2D(weights[0], (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(weights[1], (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[1], (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(weights[2], (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[2], (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[2], (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    f3 = x

    # Block 4
    x = Conv2D(weights[3], (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[3], (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(
        x)
    x = Conv2D(weights[3], (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)
    f4 = x

    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)
    # f5 = x

    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dense(1000, activation='softmax', name='predictions')(x)
    #
    # vgg = Model(img_input, x)
    # vgg.load_weights(VGG_Weights_path)

    levels = [f1, f2, f3, f4]

    o = levels[vgg_level]

    o = (Conv2D(weights[4], (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(weights[4], (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(nClasses, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)

    o2 = f3
    o2 = (Conv2D(nClasses, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(o2)

    o, o2 = crop(o, o2, img_input)

    o = Add()([o, o2])

    o = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)

    # o_shape = Model(img_input, o).output_shape
    #
    # outputHeight = o_shape[1]
    # outputWidth = o_shape[2]

    # o = (Reshape((-1, outputHeight * outputWidth)))(o)
    # o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    # model.outputWidth = outputWidth
    # model.outputHeight = outputHeight

    return model


if __name__ == '__main__':
    m = FCN8(101)
    from keras.utils import plot_model

    plot_model(m, show_shapes=True, to_file='model.png')
