import torch
from torch import nn
def comp_conv2d(conv2d, X):
    # X维度自适应
    X=X.reshape((1,1)+X.shape)
    # 卷积操作（Y储存）
    Y=conv2d(X)
    # 返回卷积结果（只返回后两个维度0123）
    return Y.reshape(Y.shape[2:])

# padding、stride
conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1,stride=1)
x=torch.randn(size=(8,8))
print(comp_conv2d(conv2d,x).shape)
# 不同高度宽度的填充
conv2d=nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=(3,4))
x=torch.randn(size=(8,8))
print(comp_conv2d(conv2d,x).shape)