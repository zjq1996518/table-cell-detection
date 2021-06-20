# 表格单元格检测

利用unet系列模型，对表格进行单元格检测  
支持的模型：unet unet++ unet3+ ma-unet  

## 说明：
+ 由于unet系列模型占用显存较大，训练数据量较少，batch_size不能设置太大，导致模型eval和train模式差距较大，于是将所有模型bn层修改为in层。 
+ 由于单元格检测任务更重要的是空间特征，而不是通道特征，将unet模型中原本初始filter由128修改为8经过测试对最终结果影响较小，且极大的压缩了模型。
+ unet中的上采样层由双线性插值更换成了转置卷积，效果有一定的提升，其他模型由于模型本身更为复杂，参数较大，没有做此改动。

## 效果：