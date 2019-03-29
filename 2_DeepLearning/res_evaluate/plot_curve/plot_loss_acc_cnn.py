# -*- coding: utf-8 -*-
'''
    把unet网络结构不同CNN卷积核个数的结果曲线画在一张图上，acc, loss
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

path = 'F:/研究僧科研/2实验结果/renew/2d_unet_cnn/'

cnn2 = open(path + 'vgg_unet_2.csv')
cnn4 = open(path + 'vgg_unet_4.csv')
cnn8 = open(path + 'vgg_unet_8.csv')
cnn16 = open(path + 'vgg_unet_16.csv')

file_cnn2 = pd.read_csv(cnn2)
file_cnn4 = pd.read_csv(cnn4)
file_cnn8 = pd.read_csv(cnn8)
file_cnn16 = pd.read_csv(cnn16)

x = file_cnn2['epoch'] + 1
x = x[0:10]
# 插值法之后的x轴值，表示从0到10间距为0.5的200个数
xnew = np.arange(min(x), max(x), 0.1)

acc_loss = [file_cnn2, file_cnn4, file_cnn8, file_cnn16]
color = ['r', 'g', 'b', 'y']
name = ['acc', 'loss']
label = ['filters num: 2', 'filters num: 4', 'filters num: 8', 'filters num: 16']

for i in range(2):
    plt.figure()
    plt.title(name[i] + ' curve')
    for j in range(4):
        al = acc_loss[j]
        y = al[name[i]].tolist()
        y = y[0:10]

        func = interpolate.interp1d(x, y, kind='cubic')

        # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
        ynew = func(xnew)
        plt.plot(xnew, ynew, color[j], label=label[j])
    plt.xlabel('epoch')
    plt.ylabel(name[i])
    plt.grid()
    plt.legend()
    plt.show()
