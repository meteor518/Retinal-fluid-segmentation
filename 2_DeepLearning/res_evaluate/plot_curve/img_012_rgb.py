# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import argparse

'''
将标签012的标记结果以RGB彩色图显示
'''
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', '-p', required= True, help='the directory of labels')
    parse.add_argument('--save-path', '-s', help="the save path of rgb labels")
    parse.add_argument('--show', default=False, help='the flag for showing the rgb label')
    args = parse.parse_args()
    
    path = args.path
    save_dir = args.save_path
    show = args.show
    dirs = os.listdir(path)

    for img_dir in dirs:
        img_list = os.listdir(path + img_dir)

        if save_dir:
            save_path = save_dir + img_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        for img_name in img_list:
            path = path + img_dir + '/' + img_name
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            m, n = np.shape(img)
            img_rgb = np.zeros((m, n, 3), dtype=np.uint8)
            img1 = img == 1
            img_r = np.uint8(img1 * 255)
            img2 = img == 2
            img_w = np.uint8(img2 * 255)
            img_rgb[:, :, 2] += img_r
            img_rgb[:, :, 2] += img_w
            img_rgb[:, :, 0] += img_w
            img_rgb[:, :, 1] += img_w

            if show:
                cv2.namedWindow('img')
                cv2.imshow('img', img_rgb)
                cv2.waitKey(0)
            if save_dir:
                cv2.imwrite(save_path + '/' + img_name, img_rgb)
                # 路径中有中文时得如下保存
                # cv2.imencode('.png', img_rgb)[1].tofile(save_img)
                # but imencode保存是按照rgb顺序保存的，所以要转为rgb保存
