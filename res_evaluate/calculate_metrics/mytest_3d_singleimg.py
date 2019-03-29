# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:09:13 2018

@author: lmx
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
    print(label.shape, pred.shape)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(pred)):
        y_pred = pred[i]
        y_label = label[i]
        for j in range(len(y_pred)):
            y_pred_temp = y_pred[j]
            y_label_temp = y_label[j]

            y_p = np.argmax(y_pred_temp, 2)
            y_l = np.argmax(y_label_temp, 2)

            # 将预测和标签的每一幅图像全部转为一列
            y_pred_list = np.reshape(y_p, (1, -1))[0].tolist()
            y_label_list = np.reshape(y_l, (1, -1))[0].tolist()
            # print(np.shape(y_pred_list))

            cm = ConfusionMatrix(y_label_list, y_pred_list)
            save_stat = cm.save_html(save_path + name + '_' + str(i) + '_' + str(j), address=False)
            save_stat == {'Status': True, 'Message': None}

            save_stat = cm.save_csv(save_path +  name + '_' + str(i) + '_' + str(j), address=False)
            save_stat == {'Status': True, 'Message': None}
