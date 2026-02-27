import torch
from torch import nn

# 单隐藏层感知机
net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
# 2个数据
x=torch.randn(size=(2,4))
print(net(x))

print(net)
print(net[2].state_dict())
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight)

# 还没做反向传播
print(net[2].weight.grad==None)
print("因为还没反向传播，所以没grad梯度")

# 一次性访问所有参数
# (name,param.shape)是要的数据的样子
# 后面是循环遍历的逻辑
# 不加星号：print([A, B]) 输出：[A, B]
# 加了星号：print(*(A, B)) 输出：A B
print(*[(name,param.shape)for name,param in net[0].named_parameters()])
print(*[(name,param.shape)for name,param in net.named_parameters()])

# net.state_dict()包含了所有权重和偏置的当前数值。
# ['2.weight']：net[2]的weight
print(net.state_dict()['2.weight'].data)
print(net.state_dict()['2.bias'])

# 从嵌套快收集参数
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
def block2():
    net=nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1())
    return net

# block就是一些层的集合
# （科研和工业中一个block承担一定的功能，类似于函数）
rgnet=nn.Sequential(block2(),nn.Linear(4,1))
print(rgnet(x))

# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        #如果是linear，就将权重初始化为均值为 0、标准差为 0.01 的高斯分布，并偏置置0
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)

# apply方法会对net遍历block
net.apply(init_normal)
print("net[0].weight.data[0]:",net[0].weight.data[0])
print("net[0].weight.data[1]:",net[0].weight.data[1])

# 同理于init_normal(m)
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print("net[0].weight.data[0]",net[0].weight.data[0])
print("net[0].bias.data[1]",net[0].bias.data[1])

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)

# 可以对不同块使用不同初始化方法
net[0].apply(xavier)
net[2].apply(init_42)
print("net[0].weight.data[0]:",net[0].weight.data[0])
print("net[2].weight.data:",net[2].weight.data)

# 直接操作的方法
net[0].weight.data[:]+=1

# 参数绑定
shared=nn.Linear(8,8)
net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
print(net)
print(net(x))

print("-----共享层检验-----")
print(net[2].weight.data[0]==net[4].weight.data[0])
net[2].weight.data[:]+=1
print(net[2].weight.data[0]==net[4].weight.data[0])

# 索引 [0]: nn.Linear(4, 8)（输入层）
#
# 索引 [1]: nn.ReLU()
#
# 索引 [2]: shared（第一次出现） —— 这是你代码里 net[2] 访问的位置。
#
# 索引 [3]: nn.ReLU()
#
# 索引 [4]: shared（第二次出现） —— 这是你代码里 net[4] 访问的位置。
#
# 索引 [5]: nn.ReLU()
#
# 索引 [6]: nn.Linear(8, 1)（输出层）