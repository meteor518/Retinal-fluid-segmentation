# -*- coding: utf-8 -*-
'''
    把4种网络结构的acc, loss等曲线画在一张图上，每张图是4个网络的曲线对比
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


if __name__ == '__main__':

    acc_loss_path = 'result/logs_csv/'

    fcn16 = open(acc_loss_path + 'vgg_fcn16.csv')
    fcn8 = open(acc_loss_path + 'vgg_fcn8.csv')
    seg = open(acc_loss_path + 'vgg_segnet.csv')
    unet = open(acc_loss_path + 'vgg_unet.csv')

    acc_loss_fcn16 = pd.read_csv(fcn16)
    acc_loss_fcn8 = pd.read_csv(fcn8)
    acc_loss_seg = pd.read_csv(seg)
    acc_loss_unet = pd.read_csv(unet)

    acc_loss = [acc_loss_fcn8, acc_loss_fcn16, acc_loss_seg, acc_loss_unet]
    color = ['r', 'g', 'b', 'y']
    label = ['fcn8', 'fcn16', 'segnet', 'unet']

    # 训练到收敛，所以每个网络的迭代次数不一致，取最大的次数，少于的取最后值补齐
    X = []
    X_len = []
    Acc = []
    Loss = []
    for i in range(4):
        x_temp = acc_loss[i]['epoch'] + 1
        x_temp = [i for i in x_temp]
        X.append(x_temp)
        X_len.append(len(x_temp))

        acc = acc_loss[i]['acc']
        acc = [i for i in acc]
        Acc.append(acc)

        l = acc_loss[i]['loss']
        l = [i for i in l]
        Loss.append(l)

    #X = np.array(X)
    index = np.argmax(X_len)

    #Acc = np.array(Acc)
    #Loss = np.array(Loss)

    # 补齐
    for i in range(4):
        temp_acc = [0] * max(X_len)
        temp_loss = [0] * max(X_len)
        if i != index:
            temp_acc[:len(Acc[i])] = Acc[i]
            temp_acc[len(Acc[i]):] = [Acc[i][-1]] * (len(temp_acc)-len(Acc[i]))

            temp_loss[:len(Loss[i])] = Loss[i]
            temp_loss[len(Loss[i]):] = [Loss[i][-1]] * (len(temp_loss)-len(Acc[i]))

            Acc[i] = temp_acc
            Loss[i] = temp_loss


    x = X[index]
    # 插值法之后的x轴值，表示从0到10间距为0.5的200个数
    xnew = np.arange(min(x), max(x), 0.1)


    plt.figure()
    plt.title('Training accuracy curve')
    for i in range(4):
        acc = Acc[i]

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
    plt.title('Training loss curve')
    for i in range(4):
        loss = Loss[i]
        # 实现函数
        func = interpolate.interp1d(x, loss, kind='cubic')

        # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
        ynew = func(xnew)
        plt.plot(xnew, ynew, color[i], label=label[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.show()
