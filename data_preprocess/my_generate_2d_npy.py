# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse

row = 496
col = 512
nClasses = 3


def create_images_data(train_path, npy_path, npy_name):
    img_list = os.listdir(train_path)
    train_imgs = np.ndarray((len(img_list), row, col, 1), dtype=np.uint8)
    i = 0

    for img_name in np.sort(img_list):
        img_path = train_path + img_name
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img.ndim == 3:
            img = img[:, :, 0]

        # 标准化
        image1 = img - np.mean(img)
        img = image1 / np.std(img)

        img = np.expand_dims(img, axis=3)
        # print(img.shape)
        train_imgs[i] = img
        if i % 100 == 0:
            print('Done: {0}/{1} train images'.format(i, len(img_list)))
        i += 1

    print("The train images' shape is ", train_imgs.shape)
    np.save(npy_path + npy_name, train_imgs)
    print('Saving to images.npy file done.')


def creat_label_data(label_path, npy_path, npy_name):
    img_list = os.listdir(label_path)
    label_imgs = np.ndarray((len(img_list), row, col, nClasses), dtype=np.uint8)
    labels = np.zeros((row, col, nClasses))
    i = 0

    for img_name in np.sort(img_list):
        path = label_path + img_name
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        if img.ndim == 3:
            img = img[:, :, 0]

        for c in range(nClasses):
            labels[:, :, c] = (img == c).astype(int)

        label_imgs[i] = labels
        if i % 100 == 0:
            print('Done: {0}/{1} label images'.format(i, len(img_list)))

        i += 1

    print('The label images shape is: ', label_imgs.shape)
    np.save(npy_path + npy_name, label_imgs)
    print('Saving to label.npy file done')


def load_data(npy_path, npy_name):
    print('-' * 30)
    print('Loading images.....')
    print('-' * 30)

    images = np.load(npy_path + npy_name)
    images = images.astype('float32')

    return images



def create_data(train_path, npy_path, npy_name):
    # 当图像在多个子目录中
    dirs = os.listdir(train_path)
    train_imgs = np.ndarray((len(dirs) * 19, row, col, 1), dtype=np.uint8)
    i = 0
    for d in dirs:
        path = train_path + d + '/'
        img_list = os.listdir(path)

        for img_name in np.sort(img_list):
            img_path = path + img_name
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            if img.ndim == 3:
                img = img[:, :, 0]
            img = np.expand_dims(img, axis=3)
            # print(img.shape)
            train_imgs[i] = img
            if i % 100 == 0:
                print('Done: {0}/{1} train images'.format(i, len(img_list)))
            i += 1

    print("The train images' shape is ", train_imgs.shape)
    np.save(npy_path + npy_name, train_imgs)
    print('Saving to images.npy file done.')

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--train-path', '-t')
    parse.add_argument('--label-path', '-l')
    parse.add_argument('--npy-path', '-n')
    parse.add_argument('--test-path', '-tst')
    parse.add_argument('--test-label', '-tl')
    args = parse.parse_args()

    train_path = os.path.abspath(os.path.expanduser(args.train_path))
    label_path = os.path.abspath(os.path.expanduser(args.label_path))
    test_path = os.path.abspath(os.path.expanduser(args.test_path))
    test_label = os.path.abspath(os.path.expanduser(args.test_label))
    npy_path = os.path.abspath(os.path.expanduser(args.npy_path))

    if train_path:
        create_images_data(train_path, npy_path, npy_name='train_sd.npy')
        creat_label_data(label_path, npy_path, npy_name='train_label.npy')
    if test_path:
        create_images_data(test_path, npy_path, npy_name='test_sd.npy')
        creat_label_data(test_label, npy_path, npy_name='tets_label.npy')



    # show figure
    # train_images = load_data(npy_path, 'train.npy')
    # label_images = load_data(npy_path, 'label.npy', label=1)
    # # test_images = load_data(npy_path, 'test.npy')
    # # test_label = load_data(npy_path, 'test_label.npy', label=1)
    #
    # for i in range(2):
    #     label = label_images[i]
    #     train_label = np.zeros(label.shape, dtype=np.uint8)
    #     train_label[:, :, 0] = label[:, :, 0] * 0 + label[:, :, 1] * 255 + label[:, :, 2] * 255
    #     train_label[:, :, 1] = label[:, :, 0] * 0 + label[:, :, 1] * 0 + label[:, :, 2] * 255
    #     train_label[:, :, 2] = label[:, :, 0] * 0 + label[:, :, 1] * 0 + label[:, :, 2] * 255
    #
    #     plt.imsave(str(i)+'.png', train_label, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    #
    #     # label = test_label[i]
    #     # test_label = label
    #     # test_label[:, :, 0] = label[:, :, 0] * 0 + label[:, :, 1] * 255 + label[:, :, 2] * 255
    #     # test_label[:, :, 1] = label[:, :, 0] * 0 + label[:, :, 1] * 0 + label[:, :, 2] * 255
    #     # test_label[:, :, 2] = label[:, :, 0] * 0 + label[:, :, 1] * 0 + label[:, :, 2] * 255
    #
    #     plt.figure()
    #     plt.subplot(121), plt.imshow(np.squeeze(train_images[i], axis=2), 'gray')
    #     plt.subplot(122), plt.imshow(train_label)
    #     plt.show()
    #
    #     # plt.figure()
    #     # plt.subplot(121), plt.imshow(np.expand_dims(test_images[i], axis=3))
    #     # plt.subplot(122), plt.imshow(test_label)
    #     # plt.imshow()
