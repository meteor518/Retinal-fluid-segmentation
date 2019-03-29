# -*- coding:utf-8 -*-
from keras.utils.vis_utils import plot_model
from keras.optimizers import *
import numpy as np
import tensorflow as tf
import time
from keras.callbacks import *
from keras import losses
import os
import shutil
import argparse

import networks.MyModel as M
from data_preprocess.my_generate_3d_npy import *
from networks.metrics import precision, recall, fmeasure
from networks.my_loss import *

np.random.seed(256)
tf.set_random_seed(256)


def train(model_name, loss_name, npy_path,
          weights_path, log_file_path, log_csv_path,
          n_classes=3, batch_size=32, epochs=100, input_size=[19, 496, 512]):
    # network model
    modelFns = {'unet_3d': M.unet_3d.get_unet, 'unet_3d_bn': M.unet_3d_bn.get_unet}
    modelFN = modelFns[model_name]
    model = modelFN(n_classes, input_size[0], input_size[1], input_size[2])
    model.summary()

    # load data
    train = load_data(npy_path, 'train_sd.npy')
    label = load_data(npy_path, 'train_label.npy')

    # model compile
    lossFns = {'ce': losses.categorical_crossentropy, 'w': W_loss, 'focal': focal_loss, 'walf': WALF_loss}
    
    model.compile(optimizer=Adam(lr=1e-3, decay=1e-6), loss=lossFns[loss_name],
                  metrics=['accuracy', precision, recall, fmeasure])

    model_name = model_name + '_' + loss_name
    model_checkpoint = ModelCheckpoint(weights_path + model_name + '.hdf5', monitor='val_acc', verbose=1,
                                       save_best_only=True)
    tb_cb = TensorBoard(log_dir=log_file_path)
    csv_logger = CSVLogger(log_csv_path + model_name + '.csv')
    early_stop = EarlyStopping(monitor='acc', patience=10)

    print('Fitting model.....')
    model.fit(train, label, batch_size=batch_size, epochs=epochs, verbose=1,
              shuffle=True, callbacks=[csv_logger, model_checkpoint, tb_cb, early_stop])


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dirs', required=True, help="The home directory of things to be saved.")
    parse.add_argument('--npy-path', required=True, help='The path od .npy files')

    parse.add_argument('--model-name', default='vgg_unet')
    parse.add_argument('--loss-name', default='ce')

    parse.add_argument('--batch-size', '-batch', type=int, default=32)
    parse.add_argument('--classes', '-c', type=int, default=3)
    parse.add_argument('--epochs', '-e', type=int, default=100)
    parse.add_argument('--size', type=int, nargs=3, default=[19, 496, 512], help='The size of input image')
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


