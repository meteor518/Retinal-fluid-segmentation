# -*- coding:utf-8 -*-
import networks.MyModel as M
from data_preprocess.my_generate_2d_npy import *
from keras.callbacks import *
import shutil
import argparse

def predict(model_name, weights_path, npy_path, n_classes=3, batch_size=32, input_height=496, input_width=512):
    modelFns = {'vgg_segnet': M.VGGSegnet.VGGSegnet, 'vgg_unet': M.VGGUnet.VGGUnet,
                'vgg_unet_bn': M.VGGUnet_bn.VGGUnet, 'vgg_fcn8': M.FCN8.FCN8, 'vgg_fcn16': M.FCN16.FCN16}

    modelFN = modelFns[model_name]
    images = load_data(npy_path, 'test_sd.npy')

    model = modelFN(n_classes, input_height=input_height, input_width=input_width)

    model.load_weights(weights_path + model_name + '.hdf5')

    preditions = model.predict(images, batch_size=batch_size, verbose=1)

    return preditions

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dirs', required=True, help="The home directory of things to be saved.")
    parse.add_argument('--npy-path', required=True, help='The path od .npy files')

    parse.add_argument('--model-name', default='vgg_unet')
    parse.add_argument('--loss-name', default='ce')

    parse.add_argument('--batch-size', '-batch', type=int, default=32)
    parse.add_argument('--classes', '-c', type=int, default=3)
    parse.add_argument('--size', type=int, nargs=2, default=[496, 512], help='The size of input image')
    args = parse.parse_args()

    batch_size = args.batch_size
    n_classes = args.classes
    input_height = args.size[0]
    input_width = args.size[1]
    model_name = args.model_name
    loss_name = '_' + args.loss_name

    model_name = model_name + loss_name

    home_path = os.path.abspath(os.path.expanduser(args.dirs))
    weights_path = home_path + 'weights/'
    npy_path = os.path.abspath(os.path.expanduser(args.npy_path))
    pred_npy = home_path + 'pred_npy/'
    if not os.path.exists(pred_npy):
        os.mkdir(pred_npy)

    imgs_mask_test = predict(model_name, loss_name, weights_path, npy_path,
                             n_classes=n_classes, batch_size=batch_size,
                             input_height=input_height, input_width=input_width)
    np.save(pred_npy + model_name + '_test_pred.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = home_path + model_name + '_preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    else:
        shutil.rmtree(pred_dir)
        os.mkdir(pred_dir)

    image_num = 0
    colors = [(0, 0, 0), (0, 0, 255), (255, 255, 255)]
    for image in imgs_mask_test:
        seg_img = np.zeros((input_height, input_width, 3))
        pr = image
        pr = pr.argmax(axis=2)

        for c in range(nClasses):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

        cv2.imwrite(os.path.join(pred_dir, 'pred_' + str("{0}".format(image_num)) + '.png'), seg_img)
        image_num += 1
