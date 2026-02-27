import torch
from setuptools.namespaces import flatten
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

net=nn.Sequential(
    # 1x224x224
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),
    # 96x54x54
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    # 96x26x26

    nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2),
    # 256x26x26
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    # 256x12x12

    nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
    # 384x12x12
    nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),
    # 384x12x12
    nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
    # 256x12x12
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    # 256x5x5

    nn.Flatten(),
    # 256x5x5=6400

    nn.Linear(6400,4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096,4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096,10),
)

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
# 模拟AlexNet对图像的需求，但其实可能是有损的
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr=0.01
num_epoch=10
d2l.train_ch6(net,train_iter,test_iter,num_epoch,lr,d2l.try_gpu())

# loss 0.330, train acc 0.880, test acc 0.878
# 1365.6 examples/sec on cuda:0