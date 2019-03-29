# 基于深度语义分割网络进行水肿分割

四个文件夹分别实现四个阶段：

## data_preprocess
数据预处理功能

* augdata.py：用于图像增强，医学图像存在大量样本不均衡，对于少数的样本增强。
```
python augdata.py -t 待增强图片路径 -l 对应标签路径 --aug-merge 保存图片和标签融合的路径 -augt 保存增强后图片路径 -augl 保存增强后的标签路径
```
* my_generate_2d_npy：病人所有图片放在一个文件夹下，将所有2d图片全部读取，大小为[图像个数, row, col]，并保存为一个.npy文件。
```
python my_generate_2d_npy.py -t 训练图片的路径 -l 训练标签路径 -n 保存.npy的路径 -tst 测试图片的路径 -tl 测试标签的路径
```

图片的目录格式如下：
>image
>>病人1_0.png
>>
>>...
>>
>>病人1_18.png
>>
>>病人2_0.png
>>
>>...
>>
>>病人2_18.png
>>
>>...依次类推

* my_generate_3d_npy：每个病人的19幅图单独存放一个文件夹，将所有文件目录下的病人图片全部读取，大小为[病人个数, 19, row, col]，并保存为一个.npy文件。
```
python my_generate_3d_npy.py -t 训练集的路径 -l 训练标签路径 -n 保存.npy的路径 -tst 测试集的路径 -tl 测试标签的路径
```
图片的目录格式如下：

>image
>>病人1
>>>0.png
>>>
>>>...
>>>
>>>18.png
>>病人2
>>>0.png
>>>
>>>...
>>>
>>>18.png
>>...依次类推

## networks
网络模型的搭建、训练、测试代码

* MyModel
该目录下为所有网络的框架搭建

* metrics.py
计算评价指标，例如acc/f1等

* my_loss.py
自己编写的各种loss函数实现代码。有weighted loss、focal loss、自己创造的多分类的WALF loss。

* my_train_2d.py / my_train_3d.py
以2D网络为例：
```训练代码
python my_train_2d.py --dirs 存储结果的主目录 -n 训练集和标签.npy文件目录 -model vgg_unet_bn -loss ce -batch 32 -c 3 -e 150
```

* my_predict_2d.py / my_predict_3d.py
```测试代码
python my_predict_2d.py --dirs 存储结果的主目录 -n 测试集.npy文件目录 -model vgg_unet_bn -loss ce -c 3
```
## res_evaluate
对结果评估的代码实现
* calculate_metrics
该文件夹下调用[pycm](https://pypi.org/project/pycm/)库，读预测结果进行各种评估系数的计算。

以2d预测结果为例
```
python mytest_2d.py -lf 测试集标签的.npy文件 -pf 测试集网络预测的.npy文件 -save 保存的评估系数文件路径 --name 预测所使用的网络模型
```
* plot_curve
画各种网络训练的acc / loss曲线图

## visualization
cnn网络中间层的结果可视化

直接运行`vgg16_intermediate_layer_visualization_gui.py`文件
