import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

# momentum 通常是一个超参数
def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    if not torch.is_grad_enabled():
    # 判断当前是否在计算梯度，如果存在说明是训练模式，否则是推理模式（eval）
    # eval阶段直接使用训练阶段算好的moving_mean 和 moving_var
        X_hat=(X-moving_mean)/torch.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4)

        if len(X.shape) == 2:
            mean=X.mean(dim=0)
            var=((X-mean)**2).mean(dim=0)
        else:
        # （batch,channel,h,w）同一channel的压缩
            mean=X.mean(dim=(0,2,3),keepdim=True)
            var=((X-mean)**2).mean(dim=(0,2,3),keepdim=True)

        X_hat=(X-mean)/torch.sqrt(var+eps)
        # 'moving_x'将当前 Batch 的统计量慢慢融合进全局统计量中，供测试时使用。
        # 1.如果想算“所有数据的精确平均”，需要记录从训练开始到现在所有 batch 的均值
        # 2.动态进化的需要
        moving_mean=momentum*moving_mean+(1-momentum)*mean
        moving_var=momentum*moving_var+(1-momentum)*var

    Y=gamma*X_hat+beta
    return Y,moving_mean.data,moving_var.data

class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        super().__init__()
        if num_dims==2:
            shape=(1,num_features)
            # 张开gamma和beta
        else:
            shape=(1,num_features,1,1)

        # nn.Parameter告诉 PyTorch：“这两个张量是可学习参数
        # 请在反向传播时为它们计算梯度，并在 optimizer.step() 时更新它们。”
        self.gamma=nn.Parameter(torch.ones(shape))
        self.beta=nn.Parameter(torch.zeros(shape))

        self.moving_mean=torch.zeros(shape)
        self.moving_var=torch.ones(shape)

    def forward(self,X):
        if self.moving_mean.device!=X.device:
        # moving_mean 默认是在 CPU 上的
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var=self.moving_var.to(X.device)
        Y,self.moving_mean,self.moving_var=batch_norm(X,self.gamma,self.beta,self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)
        print('moving_mean',self.moving_mean)
        print('moving_var',self.moving_var)
        print('gamma',self.gamma)
        print('beta',self.beta)
        return Y

# Batch应用于LeNet
net=nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=(5, 5), padding=(2, 2)),
    BatchNorm(6,num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d((2, 2)),

    nn.Conv2d(6, 16, kernel_size=(5, 5), padding=0),
    BatchNorm(16,num_dims=4),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),

    nn.Flatten(),

    nn.Linear(16 * 5 * 5, 120),BatchNorm(120,num_dims=2),nn.Sigmoid(),
    nn.Linear(120, 84),BatchNorm(84,num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10)
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

# loss 0.218, train acc 0.919, test acc 0.864
# 37994.7 examples/sec on cuda:0

# vs LeNet
# loss 0.271, train acc 0.899, test acc 0.885
# 77707.2 examples/sec on cuda:0

