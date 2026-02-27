import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 依旧人工数据集
n_train,n_test,num_inputs,batch_size=20,100,200,5
# torch.ones((num_inputs, 1)) 生成一个全是 1 的张量
true_w,true_b=torch.ones((num_inputs,1))*0.01,0.05
# 赋值语句
train_data=d2l.synthetic_data(true_w,true_b,n_train)
train_iter=d2l.load_array(train_data,batch_size)
test_data=d2l.synthetic_data(true_w,true_b,n_test)
# 测试迭代器注意加上is_train=False
test_iter=d2l.load_array(test_data,batch_size,is_train=False)

# 初始化模型参数
def init_params():
    w=torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    return [w,b]

# L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2))/2

# 训练函数
# def train(lambd):
#     w, b = init_params()
#     net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
#     num_epochs, lr = 100, 0.003
#     animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
#                             xlim=[5, num_epochs], legend=['train', 'test'])
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             with torch.enable_grad():
#                 # 增加了L2范数惩罚项，广播机制使l2_penalty(w)成为一个长度为batch_size的向量
#                 l = loss(net(X), y) + lambd * l2_penalty(w)
#             l.sum().backward()
#             d2l.sgd([w, b], lr, batch_size)
#         if (epoch + 1) % 5 == 0:
#             animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
#                                      d2l.evaluate_loss(net, test_iter, loss)))
#     print('w的L2范数是：', torch.norm(w).item())

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003

    # --- 修改点：创建列表来存储每个 epoch 的数据 ---
    animator_epochs = []
    train_loss_history = []
    test_loss_history = []

    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)

        # 每 5 个 epoch 记录一次数据（模拟原代码的画图频率）
        if (epoch + 1) % 5 == 0:
            animator_epochs.append(epoch + 1)
            train_loss_history.append(d2l.evaluate_loss(net, train_iter, loss))
            test_loss_history.append(d2l.evaluate_loss(net, test_iter, loss))

    print('w的L2范数是：', torch.norm(w).item())

    # --- 修改点：训练结束后，一次性画图 ---
    plt.figure(figsize=(8, 6))
    plt.plot(animator_epochs, train_loss_history, label='train loss')
    plt.plot(animator_epochs, test_loss_history, label='test loss', linestyle='--')
    plt.yscale('log')  # 对数坐标，和原书保持一致
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.title(f'Lambda = {lambd}')
    plt.show()  # 这行命令会弹出一个窗口显示图片


train(lambd=0)

train(lambd=3)

train(lambd=10)