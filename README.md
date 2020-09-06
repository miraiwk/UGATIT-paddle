## U-GAT-IT - Unofficial Paddle Implementation
This project is an unofficial Paddle implementation of U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation. It is adapted from  [znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch).

本项目是U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation的非官方的PaddlePaddle实现, 对官方实现[znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch)进行修改, 以适用于深度学习框架PaddlePaddle.

## 亮点
本项目**精确复现**, 所有输出数值, 所有梯度数值, 初始化权重和权重更新都与[官方版本](https://github.com/znxlwm/UGATIT-pytorch)**完全一致**.

- 通过大量的测试, 确保前向输出, 后向梯度, 权重初值分布, 优化器对权重的更新和官方版本完全一致
- 重新实现了Spectral Norm, 修复了Spectral Norm不能更新`weight_u`和`weight_v`两个状态量的Bug
- 重新实现了Instance Norm, 使用了CuDNN-Batch Norm, 计算更快, 数值更稳定
- 对PaddlePaddle 1.8.4的优化器进行了hacker, 加入了`set_lr`函数
- 重新封装的卷积, 全连接等算子, 使用Kaiming Uniform初始化卷积和全连接的权重
- 代码结构与官方实现UGATIT-pytorch一致, 并且将额外的代码放到`ops/`和`utils/`两个目录中, 对各个文件的作用进行了说明. 代码简洁, 方便阅读
- 给出了日志与关于训练过程的详细记录, 见本文档末<关于GAN训练过程的记录>
- 详细地说明了安装, 训练, 测试步骤

## 安装依赖项

本项目需要使用PaddlePaddle 1.8.4版本, 需要根据机器所安装的CUDA版本对`./requirments.txt`进行修改.

对于CUDA 9, 安装: paddlepaddle-gpu==1.8.4.post97

对于CUDA 10, 安装: paddlepaddle-gpu==1.8.4.post107

可参考[PaddlePaddle 快速安装教程](https://www.paddlepaddle.org.cn/install/quick)

修改完requirements.txt文件后, 执行:
```bash
pip install -r ./requirements.txt
```

## 准备数据
将selfie2anime数据集放到dataset目录下, 这里将数据集文件夹名称命名为data48778

使用的数据的目录格式和`YOUR_DATASET_NAME`文件夹里的目录格式保持一致.

```bash
aistudio@jupyter-8035-770177:~/work/UGATIT-paddle$ pwd
/home/aistudio/work/UGATIT-paddle
dataloader.py  dataset/       dataset.py
aistudio@jupyter-8035-770177:~/work/UGATIT-paddle$ tree dataset
dataset
├── data48778 -> /home/aistudio/data/data48778/
└── YOUR_DATASET_NAME
    ├── testA
    │   └── female_2321.jpg
    ├── testB
    │   └── 3414.png
    ├── trainA
    │   └── female_222.jpg
    └── trainB
        └── 0006.png
```

## 训练模型

这里我们训练UGATIT的Light版本的模型, Light版本的模型占用的显存比Full版本的少. 如果想训练Full版本, 可以去掉参数`--light True`.

```bash
FLAGS_cudnn_exhaustive_search=True python main.py --light True --save_freq 500 --dataset data48778 --print_freq 200
```

设置环境变量`FLAGS_cudnn_exhaustive_search`为`True`是为了PaddlePaddle选择最快的卷积算法.

`save_freq`表示模型的参数保存频率, `dataset_set`为数据集名称, `print_freq`指的是生成的图片的保存频率.

生成的结果会保存在目录`results/<数据集名称>/img`下.

如果使用selfie2anime数据集, A2B表示从真人图像变换到动漫图像, B2A表示从动漫图像到真人图像.

如果需要继续训练模型, 可以在参数后面加上`--resume True`的参数, 程序能够自动找到最近的保存点继续训练.

单卡NVIDIA Tesla V100 16G, 在selfie2anime数据集上, 256x256分辨率, Batch Size 1, 训练Light模型100万次迭代, 需要约450小时(1.6s/it). 后续会对训练速度进行优化.

## 测试模型
```bash
FLAGS_cudnn_exhaustive_search=True python main.py --light True --save_freq 500 --dataset data48778 --print_freq 200 --phase test
```
程序会加载最近的保存点, 并进行测试. 如果使用selfie2anime数据集, 通常30秒内能得到结果.

生成的结果会保存在目录`results/<数据集名称>/test`下, 每个文件对应一张图片.

如果使用selfie2anime数据集, A2B表示从真人图像变换到动漫图像, B2A表示从动漫图像到真人图像.

## 参数说明
参数名称 | 说明
---------|-----
phase|可以取值为train或test, 分别表示训练和测试, 默认为train
light|是否使用轻量模型, 默认为非轻量模型, 加上`--light True`参数后使用轻量模型
dataset|数据集的名称, 数据集放在dataset文件夹下 
iteration|迭代次数, 默认为100万次
batch_size|一次迭代的样本数量, 默认为1
print_freq|打印生成的图片的频率, 以一次迭代为单位, 默认为1000
save_freq|保存模型的频率, 以一次迭代为单位, 默认为10万. 但由于训练速度比较慢, 建议设为500或1000.
decay_flag|是否对模型的学习率进行衰减, 默认为True
lr|学习率, 默认为1e-4
weight_decay|权重衰减系数, 默认为1e-4
adv_weight|Adversarial loss的权重, 默认为1
cycle_weight|Cycle loss的权重, 默认为10
identity_weight|Identity loss的权重, 默认为10
ch|每层基础的通道数, 默认为64
n_res|Residual Block的个数, 默认为4
n_dis|判别器的层数, 默认为6
img_size|输入的图像分辨率, 默认为256
img_ch|输入的图像的通道数, 默认为3
result_dir|保存结果的目录, 默认为`results`目录
device|使用的设备, 可选cpu和cuda, 默认为cuda
resume|是否继续训练, 默认从头开始训练. 若加上`--resume True`命令, 则找到最近的保存点继续训练.


## 代码结构

- main.py

入口代码, 负责声明各种命令参数

- networks.py

U-GAT-IT模型的搭建

- UGATIT.py

U-GAT-IT模型的训练与测试

- dataset.py

对数据集的读取

- ops/

算子的实现与封装, 文件夹内有具体说明

- utils/

图像读取, 增强等实现, 文件夹内有具体说明

- logs/

训练日志


## 复现的注意事项
- 不同框架的算子权重初始化是不一样的!
- 不同框架的`model.parameters()`返回的参数列表是不一样. PaddlePaddle除了返回模型权重外, 还会返回状态量. 复现的时候可以对比参数量的数量.
- 要检查模型的输出和梯度, 让不同框架结果一致
    需要考虑到前向传播输出数值, 后向传播梯度数值, 优化器如何更新权重.
    为了方便比较两个结果, 除了对比两者的MAE, MSE外, 还可以看两个结果各自的最小值, 最大值, 均值, 方差
- 注意var函数是有偏估计还是无偏估计
- 旧版UGATIT-pytorch有两处bug:
    1. var(var(x))的形式是错的；
    2. ResnetAdaILNBlock没有加上残差

## 复现模型遇到的坑
- SpectralNorm的`weight_u`和`weight_v`不会更新, 后向传播算出的梯度有问题
- Paddle的优化器的`parameter_list`必须是列表, 如果是生成器, 不会有任何警告, 而且模型权重得不到更新
- Paddle的Linear权重是转置的
- Paddle默认不进行梯度累加, 每次backward前会对梯度清零
- 对rho参数用clip处理
- Paddle的instance norm的底层不是CuDNN-BatchNorm, 梯度是否正确以及数值稳定存疑
- 由于PaddlePaddle 1.8.4不支持训练过程中, 修改优化器的学习率(`set_lr`), 加入了`hacker_opt.py`, 以添加该函数

## 复现经验
- 首先将不同模块分开调试, 像GAN模型, 可以只训练判别器或者只训练生成器
- 用简单的小数据集训练, 让模型过拟合(不太适用于GAN)
- 检查模型的输出, 梯度, 更新后的权重是否正确
- 一定要连续测多次迭代, 否则无法发现一些状态出现bug, 如spectral norm
- 将复杂模块如ILN去掉或者换成更简单的模块, 以查找出错的地方
- 超大学习率有助于判断模型的权重是否被更新
- 衡量数组的最小值, 最大值, 均值, 方差
- 论文如果给了超参, 用论文的超参
- 残差太有用了
- 记得更新代码! 新代码可能修复了bug
- 假如有多份官方源码, 可以对比之间的不同, 如用了不同的算子, 不同的初始化

## 训练日志
[logs/UGATIT-light.log](logs/UGATIT-light.log)

数据集: selfie2anime

对各阶段的损失进行了截取, 其中`d_loss`变化范围在5以下, `g_loss`变化范围比较大, 在2000以内, 但均值有下降趋势. 只要在训练过程中, 能达到日志中的损失(只需要有一次迭代能达到), 即表明训练正常.

## 生成器和判别器
输入: `real_A`和`real_B`
```python
fake_A2B = genA2B(real_A)
fake_B2A = genB2A(real_B)

# Cycle
fake_A2B2A = genB2A(fakeA2B) # TO real_A
fake_B2A2B = genA2B(fakeB2A) # To real_B

fake_A2A = genB2A(real_A) # To real_A
fake_B2B = genA2B(real_B) # To real_B
```

- 对于genA2B, 希望输入A时输出B, 输入B时输出B
- 对于genB2A, 希望输入B时输出A, 输入A时输出A

A2B的结果图(从上到下共7张图片):
1. real_A # 原图, 真人图片
2. fake_A2A_heatmap
3. fake_A2A = genB2A(real_A) # 假设原图是动漫图片, 放到B2A的生成器, 生成真人图片
4. fake_A2B_heatmap
5. fake_A2B = genA2B(real_A) # 一般看这个就能知道生成图片的效果了, A2B中这里为生成的动漫图片
6. fake_A2B2A_heatmap
7. fake_A2B2A = genB2A(genA2B(real_A)) # 从真人图片到动漫图片, 再变回真人图片

每列对应不同的图片输入.

B2A的结果图的源域和目标域和A2B的正好反过来.

## 关于GAN训练过程的记录
- selfie2anime训练集中, 真人图片和动漫图片各3400张；测试集中, 真人图片和动漫图片各100张.
- 初始: 和原图比较相似, 真人图片逐渐磨皮, 淡化轮廓
- 前期迭代过程中, 生成的图片会出现一些黑色竖直裂缝
- 8000次迭代: 真人图片可以看到生成了动漫的眼睛
- 9750次迭代: 动漫图片可以看到真人鼻子
- 10800次迭代: 动漫图片出现真人图片的细节, 但是模糊的
- 为什么同样参数下, 滤镜颜色相似?
- 好像图片哪个区域严重变形/模糊后, 这个区域就会慢慢出现另一个域的图像, 约11450次迭代
- 真人图片容易变成动漫风格, 但动漫图片难变成真人风格
- 某时刻生成器会专注于生成某一个域的图像生成, 导致某个域的图像会变回原本的域
- 动漫人物的脸部缺少细节, 眼睛更大, 画笔更硬朗
- 13250次迭代: 真人图片转换时, 几乎所有图片的眼睛都变成动漫眼睛了
- 13350有时候会突然出现一些黑色缝隙, 而且缝隙是从上到下的, 没有水平的, 像马赛克
- 15100动漫人物开始体现出真人人脸
- 20050动漫人物看起来有立体感
- 33600有真人图片A2B比较强的美漫画风, 但动漫图片B2A怪怪的
- 同一时刻的模型会有一致的pattern
- 46000时动漫角色出现明显的真人鼻子
- 到后期, `fake_A2B2A`变得越来越像B 
- 50000时真人的脸没了, 是空的
- 61200时, 动漫角色人脸开始出现真人的整张人脸了
- 生成的图片的颜色前期比较单调, 到后期变丰富
- 会在某次迭代时模型变差, 比如84800

## 论文引用

```
@inproceedings{
Kim2020U-GAT-IT:,
title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
author={Junho Kim and Minjae Kim and Hyeonwoo Kang and Kwang Hee Lee},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJlZ5ySKPH}
}
```
