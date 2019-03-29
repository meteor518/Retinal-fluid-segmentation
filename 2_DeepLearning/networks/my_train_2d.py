# -*- coding:utf-8 -*-
from keras.utils.vis_utils import plot_model
from keras.optimizers import *
import tensorflow as tf
from keras.callbacks import *
import os
import shutil
import argparse
from keras import losses

import networks.MyModel as M
from data_preprocess.my_generate_2d_npy import *
from networks.metrics import precision, recall, fmeasure
from networks.my_loss import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

np.random.seed(256)
tf.set_random_seed(256)


def train(model_name, loss_name, npy_path,
          weights_path, log_file_path, log_csv_path,
          n_classes=3, batch_size=32, epochs=100, input_size=[496, 512]):

    # network model
    modelFns = {'vgg_segnet': M.VGGSegnet.VGGSegnet, 'vgg_unet': M.VGGUnet.VGGUnet,
                'vgg_unet_bn': M.VGGUnet_bn.VGGUnet, 'vgg_fcn8': M.FCN8.FCN8,
                'vgg_fcn16': M.FCN16.FCN16}

    modelFN = modelFns[model_name]
    m = modelFN(n_classes, input_height=input_size[0], input_width=input_size[1])
    m.summary()
    # plot_model(m, model_img, show_shapes=True)

    # load data
    train = load_data(npy_path, 'train_sd.npy')
    masks = load_data(npy_path, 'train_label.npy')

    # loss functions
    lossFns = {'ce': losses.categorical_crossentropy, 'w': W_loss, 'focal': focal_loss, 'walf': WALF_loss}

    # compile
    m.compile(loss=lossFns[loss_name], optimizer=Adam(lr=1e-3), metrics=['accuracy', precision, recall, fmeasure])

    model_name = model_name + '_' + loss_name
    model_checkpoint = ModelCheckpoint(weights_path + model_name + '.hdf5', monitor='acc', verbose=1,
                                       save_best_only=True)
    early_stop = EarlyStopping(monitor='acc', patience=8)

    tb_cb = TensorBoard(log_dir=log_file_path)
    csv_logger = CSVLogger(log_csv_path + model_name + '.csv')

    print('Fitting model...')
    m.fit(train, masks, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
          callbacks=[csv_logger, model_checkpoint, tb_cb, early_stop])


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dirs', required=True, help="The home directory of things to be saved.")
    parse.add_argument('--npy-path', '-n', required=True, help='The path od .npy files')

    parse.add_argument('--model-name', '-model', default='vgg_unet')
    parse.add_argument('--loss-name', '-loss', default='ce')

    parse.add_argument('--batch-size', '-batch', type=int, default=32)
    parse.add_argument('--classes', '-c', type=int, default=3)
    parse.add_argument('--epochs', '-e', type=int, default=100)
    parse.add_argument('--size', type=int, nargs=2, default=[496, 512], help='The size of input image')
    args = parse.parse_args()

    batch_size = args.batch_size
    n_classes = args.classes
    epochs = args.epochs
    input_size = args.size
    model_name = args.model_name
    loss_name = args.loss_name

    home_path = os.path.abspath(os.path.expanduser(args.dirs))
    weights_path = home_path + 'weights/'
    log_path = home_path + 'logs/'
    log_file_path = log_path + model_name
    log_csv_path = home_path + 'logs_csv/'

    npy_path = os.path.abspath(os.path.expanduser(args.npy_path))
    pred_npy = home_path + 'pred_npy/'

    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
        os.makedirs(log_csv_path)

    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    else:
        shutil.rmtree(log_file_path)
        os.makedirs(log_file_path)

    train(model_name, loss_name, npy_path,
          weights_path, log_file_path, log_csv_path,
          n_classes, batch_size, epochs, input_size)



