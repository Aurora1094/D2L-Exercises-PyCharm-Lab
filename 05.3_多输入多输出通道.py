import torch
from d2l import torch as d2l

# 通道融合（特征融合）【1个新特征】
def corr2d_multi_in(X,K):
    # 因为 X 和 K 的第一个维度都是“通道数”
    # zip 会把它们按通道一对一绑定起来。
    # (把二维矩阵分别对应起来)
    # 得到channel张特征图后，相同位置sum，返回二维矩阵
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))

X=torch.tensor([
    [[0,1,2],[3,4,5],[6,7,8]],
    [[1,2,3],[4,5,6],[7,8,9]]
])
K=torch.tensor([
    [[0,1],[2,3]],
    [[1,2],[3,4]]
])
print(corr2d_multi_in(X,K))

# 通道融合（特征融合）【多个新特征】
# 计算多通道输出的互相关函数
def corr2d_multi_in_out(X,K):
    # 输出k个融合特征然后stack起来
    # stack 的底层物理意义：它把松散的数据在计算机内存里强行压成了一块连续的、极其规则的内存矩阵（Tensor）
    # 这个 0 的全称是 dim=0。把这几个张量，沿着第 0 个维度（也就是最外层）叠起来
    # 最外面的中括号就是 dim=0，往里进一层就是 dim=1【这里就表示“个数/批次batch”】
    return torch.stack([corr2d_multi_in(X, k) for k in K],0)

# stack出一个卷积核
K=torch.stack((K,K+1,K+2),0)
print(K.shape)
print(corr2d_multi_in_out(X,K))

# 1x1 卷积层
def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X=X.reshape((c_i,h*w))
    K=K.reshape((c_o,c_i))
    Y=torch.matmul(K,X)
    return Y.reshape((c_o,h,w))

X=torch.normal(0,1,(3,3,3))
K=torch.normal(0,1,(2,3,1,1))

Y1=corr2d_multi_in_out_1x1(X,K)
Y2=corr2d_multi_in_out(X,K)
# 测试：证明1x1拉平算和for循环效果等价（拉平效率高）
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
print(Y1.shape)