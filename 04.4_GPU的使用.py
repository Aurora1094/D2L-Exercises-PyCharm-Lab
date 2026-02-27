import torch
from torch import nn

cpu = torch.device('cpu')
gpu = torch.device('cuda')    # 默认指向当前可用的 GPU
gpu0 = torch.device('cuda:0')  # 明确指向 0 号 GPU

# 查询当前GPU可用性以及可用GPU数量
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# 尝试获取指定的某块显卡
def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1 and torch.cuda.is_available():
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
# 获取所有可用的显卡列表
def try_all_gpus():
    device=[torch.device(f'cuda:{i}')for i in range(torch.cuda.device_count())]
    # 如果 device 列表不为空（即找到了 GPU），就返回它
    # 如果列表为空（没找到 GPU），就返回一个只包含 CPU 的列表
    return device if device else [torch.device('cpu')]
print(try_gpu())
print(try_gpu(10))
print(try_all_gpus())

# 查询张量所在设备（Tensor变量专有属性）
x=torch.tensor([1,2,3,4,5])
# y=10
print("x在",x.device)
# print("y在",y.device)
print(x)

# 使用GPU
x=torch.ones(2,3,device=try_gpu(0))
print(x.device)
print(x)

y=torch.ones(2,3,device=try_gpu(10))
print(y.device)
print(y)

# print(x+y)
# 不行！一个在cpu，一个在gpu

# 将神经网络移至(net.to)GPU0
net=nn.Sequential(nn.Linear(3,1))
net=net.to(device=try_gpu(0))
print(net)
print(net(x))

#查询神经网络第一层在哪里运行
print(net[0].weight.data.device)