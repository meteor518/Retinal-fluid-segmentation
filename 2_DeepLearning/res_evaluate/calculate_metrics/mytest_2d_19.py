# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:09:13 2018
@author: lmx

将2D网络预测的结果每个人一组的评估结果作为一个文件保存，每19张是同一个人的一组
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

    label_file = args.label_file
    pred_file = args.pred_file
    save_path = args.save_path

    name = args.name

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    pred = np.load(pred_file)
    label = np.load(label_file)
    print(pred.shape)


    for i in range(len(pred)//19):
        y_pred = []
        y_label = []
        for j in range(i*19, (i+1)*19):
            y_pred_temp = pred[j]
            y_label_temp = label[j]

            y_p = np.argmax(y_pred_temp, 2)
            y_l = np.argmax(y_label_temp, 2)


            y_p = np.reshape(y_p, (1, -1))[0].tolist()
            y_l = np.reshape(y_l, (1, -1))[0].tolist()
            y_pred.extend(y_p)
            y_label.extend(y_l)

        cm = ConfusionMatrix(y_label, y_pred)
        save_stat = cm.save_html(save_path + name + '_' + str(i), address=False)
        save_stat == {'Status': True, 'Message': None}

        save_stat = cm.save_csv(save_path + name +  '_' + str(i), address=False)
        save_stat == {'Status': True, 'Message': None}
