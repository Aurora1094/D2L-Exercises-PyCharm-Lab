import torch
from torch import nn
from d2l import torch as d2l

# 卷积操作
def corr2d(X,K):
    # X 输入图像
    # K 卷积核
    # h行、w列
    h,w=K.shape
    # Y用于存放输出结果
    # X.shape：会返回一个表示矩阵维度的元组（Tuple）（行[0]，列[1]）
    # shape[0]行；shape[1]列
    Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    # wx相乘累加
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y

X=torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
K=torch.tensor([[0,1],[2,3]])
print(corr2d(X,K))

# 实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D,self).__init__()
        self.weight=nn.Parameter(torch.rand(kernel_size))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,X):
        return corr2d(X,self.weight)+self.bias

X=torch.ones((6,8))
X[:,2:6]=0
print(X)

# 边缘检测（垂直边缘）
K=torch.tensor([[1,-1]])
# 卷积核也需要是二维的
Y=corr2d(X,K)
print(Y)
print(corr2d(X.t(),K))
print(corr2d(X.t(),K.t()))


# 学习由X生成Y的卷积核
# 定义卷积层：单通道输入输出，卷积核（1，2）无偏置
conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)

# 调整feature和label形状
X=X.reshape(1,1,6,8)
Y=Y.reshape(1,1,6,7)

for i in range(10):
    # Y_hat是feature_X的预测值
    Y_hat=conv2d(X)
    # 与label找loss
    l=(Y_hat-Y)**2
    # 每次循环清空梯度防累积（反向传播前做一下）
    conv2d.zero_grad()
    # 反向传播更新梯度
    l.sum().backward()
    # 3e-2是学习率lr（x=x-lr*grad）
    conv2d.weight.data[:]-=3e-2*conv2d.weight.grad
    if(i+1)%2==0:
        print(f'batch{i+1},loss{l.sum():.3f}')

print(conv2d.weight.data.reshape((1,2)))
