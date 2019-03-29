# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:09:13 2018
@author: lmx

将2D网络的预测出的二维图片整体进行系数评估，保存一个文件
"""
import numpy as np
import os
import argparse
from res_evaluate.calculate_metrics.pycm import *


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--label-file', '-lf', required=True, help='the .npy file of testset labels')
    parse.add_argument('--pred-file', '-pf', required=True, help='the .npy file of testset predictions')
    parse.add_argument('--save-path', '-save', required=True, help="the save directory of metrics' file")
    parse.add_argument('--name', default='vgg_unet', help='the save file name')
    args = parse.parse_args()

    label_file= args.label_file
    pred_file = args.pred_file
    save_path = args.save_path

    name = args.name

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pred = np.load(pred_file)
    label = np.load(label_file)
    print(pred.shape, label.shape)

    # 将预测的所有图像全部转为一列
    y_pred = []
    for i in range(len(pred)):
        y_temp = pred[i]
        y = np.argmax(y_temp, 2)
        yy = np.reshape(y, (1, -1))[0].tolist()
        y_pred.extend(yy)

    # 将所有图像的标签全部转为一列
    y_label = []
    for i in range(len(label)):
        y_temp = label[i]
        y = np.argmax(y_temp, 2)
        yy = np.reshape(y, (1, -1))[0].tolist()
        y_label.extend(yy)

    cm = ConfusionMatrix(y_label, y_pred)
    save_stat = cm.save_html(save_path + name, address=False)
    save_stat == {'Status': True, 'Message': None}

    save_stat = cm.save_csv(save_path + name, address=False)
    save_stat == {'Status': True, 'Message': None}
