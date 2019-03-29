# -*- coding:utf-8 -*-
import networks.MyModel as M
from data_preprocess.my_generate_3d_npy import *
from keras.callbacks import *
import os
import shutil
import argparse


def predict(model_name, weights_path, npy_path, n_classes=3, batch_size=2, input_size=[19, 496, 512]):
    test = load_data(npy_path, 'test_sd.npy')

    modelFns = {'unet_3d': M.unet_3d.get_unet, 'unet_3d_bn': M.unet_3d_bn.get_unet}
    modelFN = modelFns[model_name]
    model = modelFN(n_classes, input_size[0], input_size[1], input_size[2])

    model.load_weights(weights_path + model_name + '.hdf5')

    predictions = model.predict(test, batch_size=batch_size, verbose=1)

    return predictions


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dirs', required=True, help="The home directory of things to be saved.")
    parse.add_argument('--npy-path', '-n', required=True, help='The path od .npy files')

    parse.add_argument('--model-name', '-model', default='unet_3d')
    parse.add_argument('--loss-name', '-loss', default='ce')

    parse.add_argument('--batch-size', '-batch', type=int, default=2)
    parse.add_argument('--classes', '-c', type=int, default=3)
    parse.add_argument('--size', type=int, nargs=3, default=[19, 496, 512], help='The size of input image')
    args = parse.parse_args()

    batch_size = args.batch_size
    n_classes = args.classes
    input_size = args.size

    model_name = args.model_name
    loss_name = '_' + args.loss_name
    model_name = model_name + loss_name

    home_path = os.path.abspath(os.path.expanduser(args.dirs))
    weights_path = home_path + 'weights/'
    npy_path = os.path.abspath(os.path.expanduser(args.npy_path))
    pred_npy = home_path + 'pred_npy/'

    if not os.path.exists(pred_npy):
        os.mkdir(pred_npy)

    imgs_mask_test = predict(model_name, weights_path, npy_path, n_classes, batch_size, input_size)
    np.save(pred_npy + model_name + '_test_pred.npy', imgs_mask_test)
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    colors = [(0, 0, 0), (0, 0, 255), (255, 255, 255)]
    pred_dir = home_path + model_name + '_preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    else:
        shutil.rmtree(pred_dir)
        os.mkdir(pred_dir)

    count_processed = 0
    for x in range(0, imgs_mask_test.shape[0]):
        pred_dir1 = pred_dir + '/' + str(x) + '/'
        if not os.path.exists(pred_dir1):
            os.mkdir(pred_dir1)
        for y in range(0, imgs_mask_test.shape[1]):
            seg_img = np.zeros((input_size[1], input_size[2], 3))
            pr = imgs_mask_test[x][y]
            pr = pr.argmax(axis=2)

            for c in range(nClasses):
                seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

            cv2.imwrite(os.path.join(pred_dir1, str("{0}".format(count_processed)) + '.png'), seg_img)
            count_processed += 1

            if (count_processed % input_size[1]) == 0:
                count_processed = 0
        print('Done: {0}/{1} file images'.format(x+1, imgs_mask_test.shape[0]))

    print('-' * 30)
    print('Prediction finished')
    print('-' * 30)
