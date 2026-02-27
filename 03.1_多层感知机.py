import torch
from torch import nn
from d2l import torch as d2l

def main():

    batch_size=256
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

    #单隐藏层
    num_inputs=784
    num_outputs=10
    num_hiddens=256

    # W1：形状 (784, 256)
    # b1：形状 (256,)
    W1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True))
    b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))

    W2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True))
    b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

    #参数列表
    params=[W1,b1,W2,b2]

    def relu(x):
        a=torch.zeros_like(x)
        return torch.max(x,a)

    #m模型实现
    def net(x):
        x=x.reshape(-1,num_inputs)
        y=relu(x@W1 + b1)
        return (y@W2 + b2)

    loss=nn.CrossEntropyLoss()

    num_epochs=10
    lr=0.1

    #反向更新优化器
    updater=torch.optim.SGD(params,lr=lr)

    #训练开始

# 从 train_iter 取一批数据 (X, y)
#
# 前向：y_hat = net(X)
#
# 计算损失：l = loss(y_hat, y)
#
# 清空梯度：updater.zero_grad()
#
# 反向传播：l.backward()
#
# 参数更新：updater.step()
#
# 每个 epoch 结束去 test_iter 评估准确率

    def evaluate_accuracy(data_iter):
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in data_iter:
                y_hat = net(X)
                #(y_hat.argmax(dim=1) == y)得到布尔张量 True/False
                #例如：[True, False, True, True, ...]
                correct += (y_hat.argmax(dim=1) == y).sum().item()
                #y.numel()：y 里一共有多少元素（就是 batch_size）
                total += y.numel()
        return correct / total

    for epoch in range(num_epochs):
        total_loss, correct, total = 0.0, 0, 0

        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)

            # 每个 batch：清梯度 → 反传 → 更新
            updater.zero_grad()
            l.backward()
            updater.step()

            total_loss += l.item() * y.numel()
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.numel()

        train_loss = total_loss / total
        train_acc = correct / total
        test_acc = evaluate_accuracy(test_iter)
        print(f"epoch {epoch + 1}: loss {train_loss:.4f}, train acc {train_acc:.4f}, test acc {test_acc:.4f}")

if __name__ == "__main__":

    import torch.multiprocessing as mp
    mp.freeze_support()
    main()