# 表格单元格检测 （Pytorch）

利用unet系列模型，对表格进行单元格检测  
支持的模型：unet unet++ unet3+ ma-unet  

## 说明：
+ 由于unet系列模型占用显存较大，训练数据量较少，batch_size不能设置太大，导致模型eval和train模式差距较大，于是将所有模型bn层修改为in层。 
+ 由于单元格检测任务更重要的是空间特征，而不是通道特征，将unet模型中原本初始filter由128修改为8经过测试对最终结果影响较小，且极大的压缩了模型。
+ unet中的上采样层由双线性插值更换成了转置卷积，效果有一定的提升，其他模型由于模型本身更为复杂，参数较大，没有做此改动。

## 效果：
原图:  
![image](https://github.com/zjq1996518/table-cell-detection/blob/main/img/origin.png)  
分割结果：  
![image](https://github.com/zjq1996518/table-cell-detection/blob/main/img/mask.png)  
最终结果：
![image](https://github.com/zjq1996518/table-cell-detection/blob/main/img/result.png)  

## 总结：
+ 由于展示图中的表格内容可能涉及到公司业务，所以做了马赛克处理。
+ 本项目探究了一种比较简单的表格单元格检测的方案，本质就是训练了一个语义分割模型，背景为第0类，表格框线为第1类。
+ 训练数据是通过opencv线段检测等一系列图像算法做了初步的单元格检测，利用这些低质量的单元格做了一个mask图，然后手工筛选了约300张图作为训练数据。（如果有需要也可以开源出来。）
+ 经过测试，unet训练可以到到90%的准确率以及召回率，但还是有较低的几率出现mask断线使得找到的cell框不准确的问题。
+ 后续可能会通过该算法，再生成新的质量更好的数据，利用目标检测的算法来完成这部分工作。
