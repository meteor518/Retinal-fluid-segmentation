# -*- coding: utf-8 -*-
'''
    focal_loss的系数r与loss和acc的关系图
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

path = 'F:/研究僧科研/2实验结果/'

loss_r = open(path + 'loss_r.xlsx', 'rb')
acc_r = open(path + 'acc_r.xlsx', 'rb')

acc = pd.read_excel(acc_r)
loss = pd.read_excel(loss_r)

x = acc['epoch'] + 1
# 插值法之后的x轴值，表示从0到10间距为0.5的200个数
xnew = np.arange(min(x), max(x), 0.1)
acc_ = [acc['r=0'], acc['r=2'], acc['r=5']]
color = ['r', 'g', 'b', 'y']
label = ['r=0.5', 'r=1', 'r=2', 'r=5']

plt.figure()
# plt.title('Training accuracy curve')
for i in range(3):
    acc = acc_[i]

    # 实现函数
    func = interpolate.interp1d(x, acc, kind='cubic')

    # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
    ynew = func(xnew)
    plt.plot(xnew, ynew, color[i], label=label[i])
plt.xlabel('epoch')
plt.ylabel('acc')
plt.grid()
plt.legend()
plt.show()

plt.figure()
# plt.title('Training loss curve')
loss_ = [loss['r=0'],  loss['r=2'], loss['r=5']]
for i in range(3):
    l = loss_[i]
    # 实现函数
    func = interpolate.interp1d(x, l, kind='cubic')

    # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
    ynew = func(xnew)
    plt.plot(xnew, ynew, color[i], label=label[i])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.legend()
plt.show()
