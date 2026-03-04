import torch
from torch import nn
from d2l import torch as d2l

# 转置卷积
def trans_conv(X,K):
    h,w=K.shape
    Y=torch.zeros((X.shape[0]+h-1,X.shape[1]+w-1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]]) # 定义输入矩阵
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]]) # 定义卷积核矩阵
trans_conv(X, K)

X,K=X.reshape(1,1,2,2),K.reshape(1,1,2,2)
tconv=nn.ConvTranspose2d(1,1,2,1,0,bias=False)
tconv.weight.data=K
print(tconv(X))

tconv=nn.ConvTranspose2d(1,1,2,1,1,bias=False)
tconv.weight.data=K
print(tconv(X))

X=torch.rand(size=(1,10,16,16))
conv=nn.Conv2d(10,20,5,padding=2,stride=3)

# 转置卷积
tconv=nn.ConvTranspose2d(20,10,5,padding=2,stride=3)
print(tconv(conv(X)).shape==X.shape)

# 把一个抽象的卷积核 K 变成一个具体的线性变换矩阵 W  （都展平）
def kenernel2mat(K):
    k,W=torch.zeros(5),torch.zeros((4,9))
    # k 的内容：$[k_{00}, k_{01}, 0, k_{10}, k_{11}]$
    k[:2],k[3:5]=K[0,:],K[1,:]
    W[0,:5],W[1,1:6],W[2,3:8],W[3,4:]=k,k,k,k
    return W

W=kenernel2mat(K)
print(W)

# 验证卷积
Y = d2l.corr2d(X, K)
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))
# 验证转置卷积
Z = trans_conv(Y, K)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))