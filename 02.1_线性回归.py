from cProfile import label

import torch
x=torch.arange(4.0)
print(x)
print("--------------")
x.requires_grad_(True)#需要存储梯度
print(x)
y=2*torch.dot(x,x)
y.backward()
print(x.grad)

x.grad.zero_()#梯度清零
z=x*x
z.sum().backward()
print(x.grad)

x.grad.zero_()
u=2
t=u*x
t.sum().backward()
print(x.grad==u)

#自动求导实现
x=torch.arange(4.0)
x.requires_grad_(True)#x.grad
y=2*torch.dot(x,x)
y.backward()
print(x.grad)
#默认情况下，torch会累计梯度，需要grad_zero_()


#自定义求导
print("--------自定义求导--------")
class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        print("forward called")
        ctx.save_for_backward(x)
        return x**3

    @staticmethod
    def backward(ctx, grad_out):
        print("backward called")
        (x,) = ctx.saved_tensors
        return grad_out * 3 * x**2

#x = torch.randn(2,3,4, requires_grad=True)
x = torch.tensor([
    [[-1.00, -0.50,  0.00,  0.50],
     [ 1.00,  1.50,  2.00,  2.50],
     [ 3.00,  2.00,  1.00,  0.00]],

    [[ 0.25,  0.50,  0.75,  1.00],
     [-0.25, -0.50, -0.75, -1.00],
     [ 0.10,  0.20,  0.30,  0.40]]
], dtype=torch.float32, requires_grad=True)
print("-----输入-----")
print(x)
y = MyFunc.apply(x)         # 只打印 forward called
print("-----前向传播-----")
print(y)
L=y.sum()
L.backward()          # 才会打印 backward called
print("-----反向传播-----")
#x.grad=∂L/∂y=(∂L/∂y)*(∂y/∂x)=(1)⋅(3x^2) [梯度矩阵]
#L=∑yi
print(x.grad)


print("--------线性回归--------")
import random
import torch
#pip install d2l
from d2l import torch as d2l
import matplotlib.pyplot as plt

#构造人造数据集
def synthetic_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))#-1 表示“这一维自动推断”（这里就是 num_examples）
#reshape是为了严格保证数学上的形状

true_w=torch.tensor([2,-3.4])
true_b=4.2
#获取训练样本和标签
features,labels=synthetic_data(true_w,true_b,1000)

# #绘图
# d2l.set_figsize()
# x = features[:, 1].detach().cpu().numpy()
# y = labels.squeeze(-1).detach().cpu().numpy()
#
# d2l.plt.scatter(x, y, s=1)
# d2l.plt.show()

def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)#下标打乱，随机访问样本
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]
        #features[batch_indices]这是batch_indices索引的features的一组集合

batch_size=10
i=0
for X,y in data_iter(batch_size,features,labels):
    i+=1
    print("episode",i)
    print(X,'\n',y)

w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

#线性回归模型
def linreg(x,w,b):
    return torch.matmul(x,w) + b
#损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2
#优化算法(小批量随机梯度下降)
def sgd(params,lr,batch_size):
    with torch.no_grad():    #不再记录计算图
    #参数更新这一步是“用反向传播算出来的梯度去修正参数”
    # 它本身不是模型的“前向函数”，所以不应该被 autograd 继续追踪进计算图
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()   #不清空的话batch2的梯度会加到batch1上面

#超参数
lr=0.03
num_epochs=10
net=linreg  #相当于给函数起别名（用于对齐）
loss=squared_loss
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y)#小批量损失
        l.sum().backward()#计算w.grad 和 b.grad
        sgd([w,b],lr,batch_size)
    with torch.no_grad():#逐个遍历取平均评价效果
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean())}')


print("--------线性回归（简洁实现）--------")
import numpy as np
from torch.utils import data
true_w=torch.tensor([2,-3.4])
true_b=4.2
#生成数据
features,labels=synthetic_data(true_w,true_b,1000)
def load_array(data_arrays,batch_size,is_train=True):
    datasets=data.TensorDataset(*data_arrays)
    #data.TensorDataset是在形成数据对
    return data.DataLoader(datasets,batch_size,shuffle=is_train)
    #data.DataLoader一批一批返回,相当于yield
batch_size=10
data_iter=load_array((features,labels),batch_size)
print(next(iter(data_iter)))#打印迭代器中的第一个batch

from torch import nn
net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)#随机出初始化w
net[0].bias.data.fill_(0)

loss=nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr=0.03)#net.parameters()是net中的参数w和b

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)      # 1) 前向：算预测 + 损失
        trainer.zero_grad()      # 2) 梯度清零
        l.backward()             # 3) 反向：算梯度
        trainer.step()           # 4) 更新参数

    with torch.no_grad():
        l = loss(net(features), labels)   # 5) 每个 epoch 结束算一次全量loss
    print(f'epoch{epoch+1},loss{float(l.mean())}')
