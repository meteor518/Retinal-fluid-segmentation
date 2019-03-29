运行的主函数为: main_gray.m和main_OTSU.m

`main_OTSU.m`：

利用OTSU阈值法，直接对整张OCT图像进行二值化分割

`main_gray.m`：

先利用阈值分割法得到ILM层，再根据组织层的灰度梯度变化分割RPE层，从而将视网膜区域分割出来。

在视网膜区域内再进行阈值分割，分割出水肿。


剩余的三个函数均为调用的子函数，函数功能如下：

`anisodiff_PM.m`:各项异性扩散滤波(PM滤波)

`ILMRPE_seg.m`：得到ILM层和RPE层

`Integrity_test.m`：因为ILM/RPE层分割时，分割线可能不连续出现错位间断等现象，该函数用于纠正分割线的不连续性，使边界光滑连续。