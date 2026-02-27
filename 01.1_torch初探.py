import torch
x=torch.arange(12)
print(x.shape)
x=x.reshape(3,4)
print(x.shape)
x=torch.tensor([1,2,3,4,5])
print(x.shape)
x=torch.tensor([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
print(x.shape)
y=torch.cat([x,x],dim=0)
print(y)
z=torch.cat([x,x],dim=1)
print(z)

# 广播机制
a=torch.arange(3).reshape(1,3)
b=torch.arange(2).reshape(2,1)
print(a)
print(b)
print(a+b)

print(x)
print(x[-1])#倒数第i行
print(x[-2])
print(x[-3])
print(x[1:3])#倒数第i行到倒数第j-1行

x[-1]=1
print(x)
print("--------------")
# x=x.sum(axis=1)
# print(x)
x=x.sum(axis=1,keepdim=True)
print(x)