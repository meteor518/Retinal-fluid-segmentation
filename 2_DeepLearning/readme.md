# 基于深度语义分割网络进行水肿分割

四个文件夹分别实现四个阶段：

## data_preprocess
数据预处理功能

* augdata.py：用于图像增强，医学图像存在大量样本不均衡，对于少数的样本增强。
* my_generate_2d_npy：病人所有图片放在一个文件夹下，将所有2d图片全部读取，大小为[图像个数, row, col]，并保存为一个.npy文件。

图片的目录格式如下：

--image

----病人1_0.png

----...

----病人1_18.png

----病人2_0.png

----...

----病人2_18.png

----...依次类推

* my_generate_3d_npy：每个病人的19幅图单独存放一个文件夹，将所有文件目录下的病人图片全部读取，大小为[病人个数, 19, row, col]，并保存为一个.npy文件。

图片的目录格式如下：

--image

----病人1{
--------0.png

--------...

--------18.png}

----病人2{

--------0.png

--------...

--------18.png}

----...依次类推