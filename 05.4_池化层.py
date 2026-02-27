import torch
from torch import nn
from d2l import torch as d2l

# 手写pooling stride=1
def pool2d(X,pool_size,mode='max'):
    p_h,p_w=pool_size
    Y=torch.zeros(X.shape[0]-p_h+1,X.shape[1]-p_w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='max':
                Y[i,j]=X[i:i+p_h,j:j+p_w].max()
            elif mode=='avg':
                Y[i,j]=X[i:i+p_h,j:j+p_w].float().mean()
    return Y

X=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(X)
print(pool2d(X,(2,2),mode='max'))
print(pool2d(X,(2,2),mode='avg'))

X=torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
print(X)
# nn.MaxPool2d默认stride=kernel_size
pool2d=nn.MaxPool2d(3)
print(pool2d(X))

# 手动设定填充和步幅
pool2d=nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))

pool2d=nn.MaxPool2d((2,3),padding=(1,1),stride=(2,3))
print(pool2d(X))

# 池化层在每个输入通道上单独运算
# dim=0是batch，dim=1是同一batch的多重通道信息
X=torch.cat((X,X+1),dim=1)
print(X)
pool2d=nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))