import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 施加扰动
def dropout_layer(x,dropout):
    assert 0<=dropout<=1
    if dropout==1:
        return torch.zeros_like(x)
    if dropout==0:
        return x
    # 生成掩码
    mask=(torch.Tensor(x.shape).uniform_(0,1)>dropout).float()
    # *是矩阵逐个元素相乘（相当于有掩码1的地方做运算，掩码0的地方是0）
    return mask*x/(1.0-dropout)

#测试dropout
#torch.arange(16)生成一个从 0 开始，到 16 结束（不包含 16） 的整数序列。
x=torch.arange(16,dtype=torch.float32).reshape((2,8))
print(x)
print(dropout_layer(x,0.))
print(dropout_layer(x,0.5))
print(dropout_layer(x,1.))

num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256
# dropout1,dropout2=0.2,0.5
dropout1,dropout2=0,0

# 定义两层隐藏层的多层感知机
class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,is_train=True):
        super(Net,self).__init__()
        self.num_inputs=num_inputs
        self.training=is_train
        self.lin1 = nn.Linear(num_inputs,num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1,num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2,num_outputs)
        self.relu=nn.ReLU()

    def forward(self,x):
        H1=self.relu(self.lin1(x.reshape((-1,self.num_inputs))))
        if self.training==True:
            H1=dropout_layer(H1,dropout1)
        H2=self.relu(self.lin2(H1))
        if self.training==True:
            H2=dropout_layer(H2,dropout2)
        out=self.lin3(H2)
        return out

# 手写ch3
def evaluate_accuracy(net, data_iter, device=None):
    """计算在指定数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式 (不启用 Dropout)

    # 如果没指定设备，就用网络参数所在的设备
    if device is None:
        device = next(iter(net.parameters())).device

    metric = [0.0, 0.0]  # [预测正确的数量, 总样本数]
    with torch.no_grad():  # 评估时不需要算梯度，省内存
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)

            # 计算准确率
            y_hat = net(X)
            cmp = y_hat.argmax(axis=1) == y
            metric[0] += cmp.sum().item()
            metric[1] += y.numel()

    return metric[0] / metric[1]


import time


def train_and_plot(net, train_iter, test_iter, loss, num_epochs, updater):
    device = next(iter(net.parameters())).device
    print(f'Training on {device}')

    # 用于记录数据的列表
    epochs_list = []
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        start_time = time.time()
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            with torch.no_grad():
                train_l_sum += l.sum()
                train_acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
                n += y.numel()

        test_acc = evaluate_accuracy(net, test_iter, device)

        # 记录本轮数据
        epochs_list.append(epoch + 1)
        train_loss_list.append(train_l_sum.item() / n)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch + 1}, Loss: {train_loss_list[-1]:.4f}, '
              f'Train Acc: {train_acc_list[-1]:.3f}, Test Acc: {test_acc:.3f}, '
              f'Time: {time.time() - start_time:.1f}s')

        print("Training finished. Plotting results...")
        plt.figure(figsize=(7, 5))
        plt.plot(epochs_list, train_loss_list, label='train loss', color='blue')
        plt.plot(epochs_list, train_acc_list, label='train acc', linestyle='--', color='purple')
        plt.plot(epochs_list, test_acc_list, label='test acc', linestyle='-.', color='green')

        plt.xlabel('epoch')
        plt.legend()
        plt.grid(True)
        plt.title('Training Results (Dropout)')
        plt.show()  # 这一步会弹出一个窗口显示图片


if __name__ == '__main__':
    # 1. 实例化网络 (默认就在 CPU)
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2, True)

    # 2. 设置参数
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')  # 建议加上 reduction='none' 配合你的手动求和

    # 【注意一下，多次忽视了】3. 加载数据 (这一步在 Windows 必须放在 if __name__ 下面)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 4. 定义优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 5. 开始训练
    train_and_plot(net, train_iter, test_iter, loss, num_epochs, trainer)