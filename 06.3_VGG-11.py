import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

def vgg_block(num_convs,in_channels,out_channels):
    layers = []
    # 单纯循环num_convs次数，不加i是因为后面也用不上
    for _ in range(num_convs):
        # 当 kernel=3 且 padding=1 时，卷积后的图片高和宽不会改变
        # 保证了可以无限堆叠，而不至于让图片缩得太快。
        layers.append(nn.Conv2d
                      (in_channels,out_channels,kernel_size=3,stride=1,padding=1))
        # 每卷一次，就加一个非线性激活
        layers.append(nn.ReLU())
        # 上一次输出是下一次输入(第一轮循环会变)
        in_channels = out_channels
    # 图片的宽和高直接减半（因为步长 stride=2）
    # 这是 VGG 控制特征图尺寸的唯一手段
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    # layers解包后一次性逐个传入
    return nn.Sequential(*layers)

# 块中（卷积层数，输出通道数）
conv_arch=((1,64),(1,128),(2,256),(2,512),(2,512))

def vgg(conv_arch):
    conv_blks=[]
    in_channels=1
    for (num_convs,out_channels) in conv_arch:
        conv_blks.append(vgg_block
                         (num_convs,in_channels,out_channels))
        in_channels=out_channels

    return nn.Sequential(*conv_blks,
                         nn.Flatten(),
                         # 最后一次循环：你的 conv_arch 最后一个元素是 (2, 512)
                         # 这时，out_channels 被赋值为 512
                         # 当循环结束，代码往下执行到 nn.Linear 时
                         # 这个变量 out_channels 依然保存在内存里，其值就是 512
                         nn.Linear(out_channels*7*7,4096),
                         nn.ReLU(),
                         nn.Dropout(p=0.5),

                         nn.Linear(4096,4096),
                         nn.ReLU(),
                         nn.Dropout(p=0.5),

                         nn.Linear(4096,10)
                         )

# 缩小网络，便于快速验证跑通
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

X=torch.randn((1,1,224,224))
for blk in net:
    X=blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用 GPU 训练模型 (对应图中 Chapter 6 的定义)"""

    # 1. 初始化权重
    def init_weights(m):
        # 对全连接层和卷积层进行 Xavier 均匀分布初始化
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # 2. 准备设备与优化器
    print('training on', device)
    net.to(device)  # 将网络移动到指定的 GPU 或 CPU

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 使用随机梯度下降
    loss = nn.CrossEntropyLoss()  # 深度学习分类任务标准的交叉熵损失

    # 3. 初始化可视化对象 (PyCharm 中需要最后 plt.show() 才能显示)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    # 4. 训练迭代
    for epoch in range(num_epochs):
        # 训练损失之和, 训练准确率之和, 样本数
        metric = d2l.Accumulator(3)
        net.train()  # 确保模型处于训练模式 (启用 Dropout 和 BatchNorm)

        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 梯度清零，防止梯度累加

            # 将数据搬运到与模型相同的设备上
            X, y = X.to(device), y.to(device)

            y_hat = net(X)  # 前向传播
            l = loss(y_hat, y)  # 计算损失
            l.backward()  # 反向传播计算梯度
            optimizer.step()  # 根据梯度更新参数

            # 统计数据：不需要计算梯度以节省显存
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()

            # 更新训练进度图表
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        # 每个 epoch 结束，在测试集上评估准确率
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    # 5. 输出最终结果
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

    # --- 关键：在 PyCharm 中显示最终图表 ---
    plt.show()


batch_size=128
lr=0.05
num_epochs=10

train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())

# loss 0.174, train acc 0.935, test acc 0.923
# 1020.3 examples/sec on cuda:0