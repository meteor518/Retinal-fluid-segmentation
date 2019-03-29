# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import cv2
import argparse


def Aug(train_path, label_path, aug_merge_path):
    # 增强
    datagen = ImageDataGenerator(
        rotation_range=15,  # 旋转范围, 随机旋转(0-180)度
        width_shift_range=0.2,  # 随机沿着水平或者垂直方向，以图像的长宽小部分百分比为变化范围进行平移;
        height_shift_range=0.2,
        shear_range=0.2,  # 水平或垂直投影变换
        horizontal_flip=True,  # 水平翻转图像
        fill_mode='nearest')  # 填充像素, 出现在旋转或平移之后

    dirs = os.listdir(train_path)
    for d in dirs:
        img_list = os.listdir(os.path.join(train_path, d))
        aug_num = 0
        # 一个目录下为一个人的一组图，用同样的增强方式
        for seed in range(10, 13):
            aug_merge_dir = aug_merge_path + d + '_' + str(aug_num)
            if not os.path.exists(aug_merge_dir):
                os.makedirs(aug_merge_dir)

            for img_name in img_list:
                img_name_prefix = img_name.split('.png')[0]

                img_t = load_img(train_path + d + '/' + img_name)
                img_l = load_img(label_path + d + '/' + img_name)
                x_t = img_to_array(img_t)
                x_l = img_to_array(img_l)
                x_t[:, :, 2] = x_l[:, :, 0]
                img_tmp = array_to_img(x_t)

                img = x_t
                img = img.reshape((1,) + img.shape)

                doAugmentate(img, datagen, aug_merge_dir, img_name_prefix, seed)
            aug_num += 1
        print('The file ', d, ' is over~')


def doAugmentate(img, datagen, save_to_dir, save_prefix, seed, batch_size=1, save_format='png', imgnum=0):
    """
    augmentate one image
    """
    datagen = datagen
    i = 0
    for batch in datagen.flow(img, batch_size=batch_size, save_to_dir=save_to_dir, save_prefix=save_prefix, seed=seed,
                              save_format=save_format):
        i += 1
        if i > imgnum:
            break


def splitMerge(path_merge, path_train, path_label):
    """
    split merged image apart
    """
    merge_dirs = os.listdir(path_merge)
    for d in merge_dirs:
        path = path_merge + d
        save_train_path = path_train + d
        save_label_path = path_label + d
        if not os.path.exists(save_label_path):
            os.makedirs(save_label_path, exist_ok=True)
        if not os.path.exists(save_train_path):
            os.makedirs(save_train_path, exist_ok=True)

        img_list = os.listdir(path)
        for img_name in img_list:
            # print(os.path.join(path, img_name))
            img = cv2.imdecode(np.fromfile(path + '/' + img_name, dtype=np.uint8), -1)
            # print(np.shape(img))
            img_train = img[:, :, 2]  # cv2 read image:bgr; load_img:rgb
            img_label = img[:, :, 0]
            cv2.imencode('.png', img_train)[1].tofile(save_train_path + '/' + img_name)
            cv2.imencode('.png', img_label)[1].tofile(save_label_path + '/' + img_name)

        print(d, ' is over~')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--train-path', '-t', required=True)
    parse.add_argument('--label-path', '-l', required=True)
    parse.add_argument('--aug-merge', required=True)
    parse.add_argument('--aug-train', '-augt', required=True)
    parse.add_argument('--aug-label', '-augl', required=True)
    args = parse.parse_args()

    train_path = os.path.abspath(os.path.expanduser(args.train_path))
    label_path = os.path.abspath(os.path.expanduser(args.label_path))
    aug_merge_path = os.path.abspath(os.path.expanduser(args.aug_merge))
    aug_train = os.path.abspath(os.path.expanduser(args.aug_train))
    aug_label = os.path.abspath(os.path.expanduser(args.aug_label))

    print('-' * 30)
    print('Start augdata....')
    Aug(train_path, label_path, aug_merge_path)
    print('All files aug is done')

    print('-' * 30)
    print('Start split merge....')
    splitMerge(aug_merge_path, aug_train, aug_label)
