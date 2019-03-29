# 基于模糊C均值聚类法进行OCT水肿分割
最终主代码为`main_sfcm.m`。

## main_sfcm.m
* 修改代码中图片路径，直接运行该文件即可。代码是采用基于FCM修改的sfcm函数进行聚类分割。sfcm函数公式见[本人论文](https://ieeexplore.ieee.org/abstract/document/7603476/)。

* 代码中采用两次SFCM分割，第一次整体聚类为了得到ILM层，第二次在视网膜区域SFCM分割水肿。

## main_fcm.m
代码直接调用MATLAB工具库的FCM函数，利用传统fcm分割。

## main_km_clustersnum.m
该文件代码是给定不同的初始聚类数，使用kmeans聚类，利用轮廓系数评价聚类结果，从而确定聚类的类别数。

        当确定聚类类别数时，该文件不用考虑，如果不确定类别数，可以考虑使用该方法确定。

## 剩下其他函数均为调用的子函数
`anisodiff_PM.m`用于滤波； `ILMRPE_seg`得到ILM层和RPE层； `Integrity_test.m`分割线完整性检测。

`initfcm.m`、`disfcm.m`、`stepfcm.m`、`sfcm.m`：是根据fcm函数库，自己编写的代码实现SFCM算法。
