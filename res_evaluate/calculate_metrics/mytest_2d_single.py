# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:09:13 2018
@author: lmx

将2D网络预测的结果每个人与标签进行对比，每个人的评估结果作为一个文件保存
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

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pred = np.load(pred_file)
    label = np.load(label_file)
    print(pred.shape)

    for i in range(len(pred)):
        y_pred_temp = pred[i]
        y_label_temp = label[i]

        y_p = np.argmax(y_pred_temp, 2)
        y_l = np.argmax(y_label_temp, 2)

        y_pred = np.reshape(y_p, (1, -1))[0].tolist()
        y_label = np.reshape(y_l, (1, -1))[0].tolist()

        cm = ConfusionMatrix(y_label, y_pred)
        save_stat = cm.save_html(save_path + name + '_' + str(i), address=False)
        save_stat == {'Status': True, 'Message': None}

        save_stat = cm.save_csv(save_path + name + '_' + str(i), address=False)
        save_stat == {'Status': True, 'Message': None}
