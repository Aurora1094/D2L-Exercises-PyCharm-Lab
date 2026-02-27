import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 通常我们处理的数据集（如 MNIST 手写数字）
# 在磁盘上的存储方式或经过某些预处理后
# 会变成一种“展平”的格式。
class Reshape(torch.nn.Module):
    def forward(self, X):
        # .view()也是reshape的一种方式，但不改变数据本身，和reshape不同
        return X.view(-1,1,28,28)

# 定义网络
net=torch.nn.Sequential(
    Reshape(),
# 第一阶段
    nn.Conv2d(1,6,kernel_size=(5,5),padding=(2,2)),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),

    nn.Conv2d(6,16,kernel_size=(5,5),padding=0),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),

# 过渡阶段
    nn.Flatten(),# 拉平操作：进入全连接层前必须要做！！

# 第二阶段
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10)
)

# 检查模型
X=torch.randn(2,1,28,28).float() # 两张图片
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

# 测试LeNet在Fasion-MNIST数据集上的表现
batch_size=256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

# 改写evaluate_accuracy,适应gpu
def evaluate_accuracy(net, data_iter,device=None):
    if isinstance(net, torch.nn.Module):
    # 检查 net 是否为 PyTorch 的模型类
        net.eval()
        # 将模型设置为评估模式(关闭dropout等)
        if not device:
            device=next(iter(net.parameters())).device
            # 通过查看模型第一个参数所在的设备
            # 来自动推断模型目前是在 CPU 还是 GPU 上
    metric=d2l.Accumulator(2)
    # 创建一个累加器，用于存储两个值：正确预测的数量总和和样本总数
    for X,y in data_iter:
        if isinstance(X,list):
        # 如果输入 X 是一个列表（例如在某些多输入模型中），则遍历列表。
            X=[x.to(device) for x in X]
        else:
            X=X.to(device)
        y=y.to(device)
        # y.numel()：获取当前批次的样本总数（张量的元素个数）
        # 这里是分别统计正确数与总数
        metric.add(d2l.accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

# ai
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


# 训练和评估
lr=0.9
num_epochs=10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

