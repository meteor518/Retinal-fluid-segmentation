# -*- coding: utf-8 -*-
'''
    把每种网络结构的4个曲线画在一张图上，画四个网络的图，每张是单个网络自身的
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


if __name__ == '__main__':
    unet_2d = open('result_2d/logs_csv/vgg_unet.csv')
    unet_2d_bn = open('result_2d/logs_csv/vgg_unet_bn_WA.csv')
    unet_3d = open('result_3d/logs_csv/unet_3d_bn.csv')

    unet1 = pd.read_csv(unet_2d)
    unet2 = pd.read_csv(unet_2d_bn)
    unet3 = pd.read_csv(unet_3d)

    acc_loss = [unet1, unet2, unet3]

    X = []
    X_len = []
    Acc = []
    Loss = []
    for i in range(3):
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

    index = np.argmax(X_len)

    # 补齐
    for i in range(3):
        temp_acc = [0] * max(X_len)
        temp_loss = [0] * max(X_len)
        if i != index:
            temp_acc[:len(Acc[i])] = Acc[i]
            temp_acc[len(Acc[i]):] = [Acc[i][-1]] * (len(temp_acc) - len(Acc[i]))

            temp_loss[:len(Loss[i])] = Loss[i]
            temp_loss[len(Loss[i]):] = [Loss[i][-1]] * (len(temp_loss) - len(Acc[i]))

            Acc[i] = temp_acc
            Loss[i] = temp_loss

    x = X[index]
    # 插值法之后的x轴值，表示从0到10间距为0.5的200个数
    xnew = np.arange(min(x), max(x), 0.1)


    color = ['r', 'g', 'b', 'y']
    label = ['2D U-Net', '2D-WALF-UNet', '3D-WALF-UNet']

    plt.figure()
    plt.title('Training accuracy curve')
    for i in range(3):
        acc =Acc[i]

        # 实现函数
        func = interpolate.interp1d(x, acc, kind='cubic')

        # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
        ynew = func(xnew)

        plt.plot(xnew, ynew, color[i], label=label[i])
    plt.xlabel('epoch')
    plt.ylabel('acc')
    # plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Training loss curve')
    for i in range(3):
        loss = Loss[i]
        # 实现函数
        func = interpolate.interp1d(x, loss, kind='cubic')

        # 利用xnew和func函数生成ynew,xnew数量等于ynew数量
        ynew = func(xnew)

        plt.plot(xnew, ynew, color[i], label=label[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.grid()
    plt.legend()
    plt.show()

