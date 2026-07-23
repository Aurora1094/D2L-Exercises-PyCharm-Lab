# D2L Exercises PyCharm Lab

《动手学深度学习》PyTorch 代码复现与练习。

本仓库使用 PyCharm 编写，将书中的主要知识点整理为独立的 Python 脚本。代码中保留了较多中文注释、运行输出和实现过程，方便在学习时逐步调试和理解。

## 目录

- [PyTorch 与基础模型](#pytorch-与基础模型)
- [卷积神经网络](#卷积神经网络)
- [计算机视觉](#计算机视觉)
- [序列模型](#序列模型)
- [注意力机制](#注意力机制)
- [Kaggle 练习](#kaggle-练习)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [运行方式](#运行方式)
- [注意事项](#注意事项)
- [学习笔记](#学习笔记)

## 已完成内容

### PyTorch 与基础模型

- 张量操作、广播、索引和自动求导
- 线性回归与 Softmax 回归
- 多层感知机
- 多项式拟合、权重衰减和 Dropout
- 参数管理、自定义层与块、模型读写和 GPU 使用

### 卷积神经网络

- 二维卷积、填充、步幅、通道和池化
- LeNet、AlexNet、VGG、NiN、GoogLeNet 和 ResNet
- 批量归一化、数据增广和迁移学习

### 计算机视觉

- 边界框与锚框
- SSD 目标检测
- 语义分割与转置卷积
- FCN 和神经风格迁移

### 序列模型

- 序列数据与文本预处理
- 语言模型和数据集构造
- 从零实现 RNN
- 编码器与解码器

### 注意力机制

- Nadaraya-Watson 核回归
- 注意力打分函数
- Bahdanau 注意力
- 自注意力与位置编码

### Kaggle 练习

`Competitions/` 目录中包含：

- 房价预测
- 树叶分类
- Tiny CIFAR-10 图像分类

## 项目结构

```text
01.*                 PyTorch 基础
02.* - 04.*          基础模型与神经网络组件
05.* - 06.*          卷积神经网络
08.*                 计算机视觉
09.*                 序列模型与 RNN
10.*                 注意力机制
Competitions/        Kaggle 练习
```

## 环境依赖

主要使用以下 Python 库：

```bash
pip install torch torchvision d2l matplotlib numpy pandas scikit-learn networkx
```

不同 PyTorch、TorchVision 和 `d2l` 版本之间可能存在接口差异，建议根据本机 CUDA 环境选择对应的 PyTorch 版本。

## 运行方式

在 PyCharm 中打开仓库后，可以直接运行单个脚本，也可以在终端执行：

```bash
python 02.1_线性回归.py
python 06.7_ResNet.py
python 09.4_RNN.py
```

## 注意事项

- 部分脚本会通过 `d2l` 自动下载数据集，需要保持网络连接。
- 目标检测、分割和风格迁移等脚本可能需要额外的图片或数据集文件。
- Kaggle 练习需要自行准备对应比赛数据。
- 少量代码中保留了本地绝对路径，运行前需要改为自己的数据目录。
- 训练 CNN、目标检测和序列模型时建议使用支持 CUDA 的 GPU。

## 学习笔记

实时笔记：[Notion](https://humane-carp-561.notion.site/2f25bc418f4d80f3895cfb390d766fb1?source=copy_link)
