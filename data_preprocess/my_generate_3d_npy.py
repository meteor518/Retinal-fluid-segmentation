# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


image_rows = int(496)
image_cols = int(512)
image_depth = 19
nClasses = 3


def create_images_data(image_path, npy_path, name):
    dirs = os.listdir(image_path)
    dirs_num = int(len(dirs))

    imgs = np.ndarray((dirs_num, image_depth, image_rows, image_cols, 1), dtype=np.uint8)  # 用于保存训练的所有样本，每一行为一个文件夹19幅图

    i = 0
    print('-' * 30)
    print('Creating images...')
    print('-' * 30)
    for dirr in np.sort(dirs):  # 所有目录，dirr为每一个目录
        j = 0
        dirr = os.path.join(image_path, dirr)
        images = np.sort(os.listdir(dirr))  # 每一个目录下的所有图像名字
        count = dirs_num
        for image_name in images:
            img = cv2.imdecode(np.fromfile(os.path.join(dirr, image_name), dtype=np.uint8), -1)  # 读取每一幅图
            if img.ndim == 3:
                img = img[:, :, 0]

            # 标准化
            image1 = img - np.mean(img)
            img = image1 / np.std(img)

            imgs[i, j] = np.expand_dims(img, axis=3)  # 保存到imgs中
            j += 1
            if j % (image_depth) == 0:
                j = 0
                i += 1
                print('Done: {0}/{1} 3d images'.format(i, count))

    print('Loading of image data done.')
    np.save(npy_path + name, imgs)
    print('Saving to %s files done.' % name)


def create_label_data(label_path, npy_path, name):
    dirs = os.listdir(label_path)
    dirs_num = int(len(dirs))

    imgs_mask = np.ndarray((dirs_num, image_depth, image_rows, image_cols, nClasses),
                           dtype=np.uint8)  # 用于保存训练的所有样本的标签，每一行为一个文件夹19幅图

    i = 0
    for dirr in np.sort(dirs):
        j = 0
        dirr = os.path.join(label_path, dirr)
        images = np.sort(os.listdir(dirr))
        count = dirs_num
        for mask_name in images:
            img_mask = cv2.imdecode(np.fromfile(dirr + '/' + mask_name, dtype=np.uint8), -1)
            if img_mask.ndim == 3:
                img_mask = img_mask[:, :, 0]

            seg_labels = np.zeros((image_rows, image_cols, nClasses))
            for c in range(nClasses):
                seg_labels[:, :, c] = (img_mask == c).astype(int)

            imgs_mask[i, j] = seg_labels
            j += 1
            if j % (image_depth) == 0:
                j = 0
                i += 1
                print('Done: {0}/{1} mask 3d images'.format(i, count))

    print('Loading of label done.')
    np.save(npy_path + name, imgs_mask)
    print('Saving to %s files done.' % name)


def load_data(npy_path, name):
    images = np.load(npy_path + name)
    images = images.astype('float32')

    return images


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
        create_images_data(train_path, npy_path, 'train_sd.npy')
        create_label_data(label_path, npy_path, 'train_label.npy')
    if test_path:
        create_images_data(test_path, npy_path, 'test_sd.npy')
        create_label_data(test_label, npy_path, 'test_label.npy')


    # show figures
    # train = load_data(npy_path, 'train.npy')
    # label = load_data(npy_path, 'label.npy', label=1)
    # test = load_data(npy_path, 'test.npy')
    # test_label = load_data(npy_path, 'test_label.npy', label=1)
    #
    # imgs_train = preprocess_squeeze(train)[0]
    # print(np.shape(imgs_train))
    # imgs_mask_train = label[0]
    # print(np.shape(imgs_mask_train))
    #
    # colors = [(0, 0, 0), (0, 0, 255), (255, 255, 255)]
    # for i in range(19):
    #     plt.figure()
    #     plt.subplot(221), plt.imshow(imgs_train[i])
    #
    #     pr = imgs_mask_train[i]
    #     pr = pr.argmax(axis=2)
    #     print(pr.shape)
    #     seg_img = np.zeros((image_rows, image_cols, 3))
    #     for c in range(nClasses):
    #         seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
    #         seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
    #         seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    #
    #     plt.subplot(222), plt.imshow(seg_img)
    #     plt.show()
    #
