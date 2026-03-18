import torch
from torch import nn
from d2l import torch as d2l

# 生成数据集
n_train=50
x_train,_=torch.sort(torch.rand(n_train)*5)

def f(x):
    return 2*torch.sin(x)+x**0.8

y_train=f(x_train)+torch.normal(0.0,0.5,(n_train,))
x_test=torch.arange(0,5,0.1)
y_truth=f(x_test)
n_test=len(x_test)
print(n_test)

def plot_kernel_reg(y_hat):
    d2l.plot(x_test,[y_truth,y_hat],'x','y',legend=['Truth','Pred'],xlim=[0,5],ylim=[-1,5])
    d2l.plt.plot(x_train,y_train,'o',alpha=.5)

y_hat=torch.repeat_interleave(y_train.mean(),n_test)
plot_kernel_reg(y_hat)
d2l.plt.show()

# 非参数注意力汇聚
# 这个矩阵的每一行，都是同一个 Query 复制了 50 份
# 这样就可以直接和 50 个 Key 进行一次性的减法运算
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# （Query-Key）用softmax直接实现归一化
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# alpha x value
y_hat = torch.matmul(attention_weights, y_train)

plot_kernel_reg(y_hat)
d2l.plt.show()

d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),xlabel='Sorted training inputs',ylabel='Sorted training inputs')
d2l.plt.show()

# 带参量的注意力汇聚

# 模拟计算
X=torch.ones((2,1,4))
Y=torch.ones((2,4,6))
# bmm是指每个batch的矩阵做乘法：（1，4）x（4，6）=（1，6）
# bmm 只接受 3 维张量
print(torch.bmm(X,Y).shape)

weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
# unsqueeze(1) 就是在第 1 个位置（索引从0开始）插入维度，变成了 (50, 1, 49)
# (50, 1, 49) × (50, 49, 1)
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 定义可训练参数，初始为随机值
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # 把Query变成Key的形状
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # self.w用于自我调整，即“可训练”
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

# 这 50 个已知数据复制 50 行，变成一个 (50, 50) 的矩阵
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
# torch.eye(n_train)：生成一个 (50, 50) 的单位矩阵。
# 对角线上全是 1，其他全是 0。这根对角线，完美对应了每个 Query 对应的“自己”
# 之后用1-tensor,转成布尔矩阵（掩码）
# X_tile[...]：用这个 True/False 的掩码去盖在 X_tile 上
# 所有是 False 的（它自己）全被扔掉了，重整为（50，49）
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(10):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

# 测试阶段，不需要剔除自己了，因为测试数据本来就不在训练数据库里
keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
# 做预测
y_hat = net(x_test, keys, values).unsqueeze(1).detach()

plot_kernel_reg(y_hat)
d2l.plt.show()

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
d2l.plt.show()