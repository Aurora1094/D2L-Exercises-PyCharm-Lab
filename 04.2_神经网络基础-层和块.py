import torch
from torch import nn
from torch.nn import functional as F

# 模型构造

# 简易写法：输入层+激活函数+输出层
net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

print("-----简易实现-----")
x=torch.rand(2,20)
print(net(x))


# 自定义写法：自定义块
class MLP(nn.Module):  # 继承 nn.Module 基类
    def __init__(self):
        super().__init__()
        # 在这里定义“零件”
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        # 在这里定义“组装流程”
        x = self.hidden(x)  # 20 -> 256
        x = F.relu(x)  # 对 256 个输出做激活 (一次就够了)
        return self.out(x)  # 256 -> 10


net = MLP()
net(x)

# 假数据
print("-----自定义块-----")
x = torch.randn(1, 20)
output = net(x)
print(output)
print(net)

# 顺序块：“net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))”
# 的展开（平时其实不用），使用简易实现和自定义块（实现复杂逻辑）更多
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # 使用 enumerate 来获取每个 block 的序号
        for idx, block in enumerate(args):
            # PyTorch 规定 _modules 的 key 必须是字符串！
            self._modules[str(idx)] = block

    def forward(self, x):
        # 遍历所有注册好的层
        for block in self._modules.values():
            x = block(x)
        return x

# 实例化网络
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

# 假数据
print("-----顺序块-----")
x = torch.randn(1, 20)
output = net(x)
print(output)
print(net)

# 用自定义块实现顺序块不具备的逻辑
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        #构造函数，随机得到一个权重，然后输入20输出20，无激活函数
        super().__init__()
        # 【非常重要！！！】rand_weight和self.linear 没有任何关系。它只是一个纯粹的、普通的 torch.Tensor
        self.rand_weight=torch.rand((20,20),requires_grad=False)
        # bias没有指定为False，会随机初始一个
        self.linear=nn.Linear(20,20)
    def forward(self, x):
        # 放进构造函数的层 x=xw{T}+b
        x=self.linear(x)
        # relu（xw（rand_weight）+1）覆盖旧状态
        x=F.relu(torch.mm(x,self.rand_weight)+1)
        # 再来一次 x=xw{T}+b
        x=self.linear(x)

        # x.abs() 是把 x 里所有数字变成绝对值
        # 【这里的逻辑是函数无法实现的】
        while x.abs().sum()>1:
            x/=2
        return x.sum().view(1, 1)

net = FixedHiddenMLP()
# net是问神经层有谁，而for、relu属于逻辑
# rand_weight是类下的成员变量，和神经层无关
print("-----FixedHiddenMLP-----")
print(net)
print(net(x))

# 混合搭配各种组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self, x):
        return self.linear(self.net(x))
print("-----NestMLP-----")
chimera=nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP(),nn.Linear(1,10))
print(chimera)
print(chimera(x))

# 个人理解：
# FixedHiddenMLP是经过层后再加入一些自己的操作（有点像自己定义了一种激活函数）
# chimera是多个层（既有官方，也有自己写的）的组合

print("下面是更底层的，自定义“层（layer）【如，linear就是层函数的一种】")

#自定义层

# 无参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x-x.mean()

print("-----CenteredLayer-----")
layer=CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

# 将层作为组件合并到更复杂的模型
net=nn.Sequential(nn.Linear(8,128),CenteredLayer())
y=net(torch.randn(4,8))
print(y)
print(y.mean())

# 含参的层
class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(in_units,units))
        self.bias=nn.Parameter(torch.randn(units))
    def forward(self, x):
        linear=torch.matmul(x,self.weight)+self.bias
        # 自带激活：后面就不需要再额外加 nn.ReLU()
        return F.relu(linear)
dense=MyLinear(5,3)#5->3的层
print(dense.weight)
y=dense(torch.randn(2,5))
print(y)