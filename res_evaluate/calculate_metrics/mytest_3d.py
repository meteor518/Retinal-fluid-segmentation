# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:09:13 2018

@author: lmx

将3D网络的预测出的整体进行系数评估，保存一个文件
"""
import numpy as np
from res_evaluate.calculate_metrics.pycm import *
import os
import argparse


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--label-file', '-lf', required=True, help='the .npy file of testset labels')
    parse.add_argument('--pred-file', '-pf', required=True, help='the .npy file of testset predictions')
    parse.add_argument('--save-path', '-save', required=True, help="the save directory of metrics' file")
    parse.add_argument('--name', default='vgg_unet', help='the save file name')
    args = parse.parse_args()

    label_file = args.label_file
    pred_file = args.pred_file
    save_path = args.save_path

    name = args.name


    pred = np.load(pred_file)
    label = np.load(label_file)
    print(label.shape)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 将预测的所有图像全部转为一列
    y_pred = []
    for i in range(len(pred)):
        y1 = pred[i]
        for j in range(len(y1)):
            y_temp = y1[j]
            y = np.argmax(y_temp, 2)
            yy = np.reshape(y, (1, -1))[0].tolist()
            y_pred.extend(yy)

    # 将所有图像的标签全部转为一列
    y_label = []
    for i in range(len(label)):
        y1 = label[i]
        for j in range(len(y1)):
            y_temp = y1[j]
            y = np.argmax(y_temp, 2)
            yy = np.reshape(y, (1, -1))[0].tolist()
            y_label.extend(yy)

    cm = ConfusionMatrix(y_label, y_pred)
    save_stat = cm.save_html(save_path + name, address=False)
    save_stat == {'Status': True, 'Message': None}

    save_stat = cm.save_csv(save_path + name, address=False)
    save_stat == {'Status': True, 'Message': None}
