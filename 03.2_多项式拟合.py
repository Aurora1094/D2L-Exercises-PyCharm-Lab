import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
#通过多项式施加噪音生成人工数据集
max_degree=20
n_train,n_test=100,100
true_w=np.zeros(max_degree)
#$$[1, 4（20）]$$
true_w[0:4]=np.array([5,1.2,-3.4,5.6])

#生成200x1数据
features=np.random.normal(size=(n_train+n_test,1))
np.random.shuffle(features)
#把200x1张成200x20的poly_feature
poly_features=np.power(features,np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    poly_features[:,i]/=math.gamma(i+1)
#Feature x w
#当 np.dot(矩阵, 一维向量) 时，NumPy 会自动把w当作4x1
labels=np.dot(poly_features,true_w)
#+noise
labels+=np.random.normal(scale=0.1,size=labels.shape)

#评估损失
def evaluate_loss(net,data_iter,loss):
    metric=d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            # metric 是一个累加器对象。
            # .add 方法会把参数 A 加到第一个变量上
            # 参数 B 加到第二个变量上。
            metric.add(l.sum(), l.numel())

    return metric[0]/metric[1]

def train(train_features, test_features, train_labels, test_labels, num_epochs=400,title=""):
    # 1. 定义损失函数
    # 注意：这里通常使用 reduction='none'，因为后面要手动 l.sum()
    loss = nn.MSELoss(reduction='none')

    # 2. 定义网络
    input_shape = train_features.shape[-1]
    # bias=False 是因为多项式特征的第一列已经是 1 (x^0)，所以不需要额外的偏置项
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))

    # 3. 准备数据迭代器
    batch_size = min(10, train_labels.shape[0])
     # 先把 numpy 转成 tensor，并指定 float32 类型（非常重要，否则会报 Double/Float 类型不匹配）
    train_features = torch.tensor(train_features, dtype=torch.float32)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

     # 然后再传给 d2l.load_array
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)),
                               batch_size, is_train=False)
    # 4. 定义优化器 (SGD)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    # 用于记录 loss 画图
    train_loss_history = []
    test_loss_history = []
    epochs = []

    print(f"=== 开始训练: {title} ===")

    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)

            # 【修改点 2】使用 mean() 而不是 sum()
            # 20维特征时 sum() 会导致梯度爆炸变成 NaN
            l.mean().backward()

            trainer.step()

        if epoch == 0 or (epoch + 1) % 20 == 0:
            # 使用我们要自己定义的 evaluate_loss
            train_l = evaluate_loss(net, train_iter, loss)
            test_l = evaluate_loss(net, test_iter, loss)

            train_loss_history.append(train_l)
            test_loss_history.append(test_l)
            epochs.append(epoch + 1)

    print(f'最终权重 (前4个): {net[0].weight.data.numpy()[0][:4]}')

    # 【修改点 3】 使用 Matplotlib 绘图并显示
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss_history, label='Train')
    plt.plot(epochs, test_loss_history, label='Test', linestyle='--')
    plt.yscale('log')  # 对数坐标看 loss 变化更清晰
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log scale)')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()  # 弹窗显示图片

    # # 5. 设置可视化 (动画图表)
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
    #                         xlim=[1, num_epochs], ylim=[1e-3, 1e2],
    #                         legend=['train', 'test'])
    #
    # # 6. 训练循环
    # for epoch in range(num_epochs):
    #     # 训练一个 epoch (这里面包含了前向传播、反向传播、更新参数)
    #     #d2l.train_epoch_ch3(net, train_iter, loss, trainer)
    #     net.train()
    #     for X, y in train_iter:
    #         trainer.zero_grad()  # 1. 梯度清零
    #         l = loss(net(X), y)  # 2. 计算损失 (Forward)
    #         l.sum().backward()  # 3. 反向传播 (Backward)
    #         trainer.step()  # 4. 更新权重 (Update)
    #
    #     # 每 20 个 epoch 记录一次并在图上画点
    #     if epoch == 0 or (epoch + 1) % 20 == 0:
    #         animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
    #                                  d2l.evaluate_loss(net, test_iter, loss)))
    #
    # # 7. 打印最终学到的权重
    # print('weight:', net[0].weight.data.numpy())



# 正确
train(poly_features[:n_train, :4],
      poly_features[n_train:, :4],
      labels[:n_train],
      labels[n_train:])

# 欠拟合
train(poly_features[:n_train, :2],
      poly_features[n_train:, :2],
      labels[:n_train],
      labels[n_train:])

# 过拟合
num=50
train(poly_features[:num, :20],
      poly_features[n_train:, :20],
      labels[:num],
      labels[n_train:])