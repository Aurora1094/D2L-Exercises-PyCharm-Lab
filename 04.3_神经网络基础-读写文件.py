import torch
from torch import nn
from torch.nn import functional as F

x=torch.arange(4)
torch.save(x,'x-file')
x2=torch.load('x-file')
print(x)
print(x2)

# 储存一个张量列表
y=torch.zeros(4)
torch.save([x,y],'x-file')
x2,y2=torch.load('x-file')
print((x2,y2))

# 读写从字符串映射到张量的字典
mydict={'x':x,'y':y}
torch.save(mydict,'mydict')
mydict2=torch.load('mydict')
print(mydict2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 字典形式保存参数
torch.save(net.state_dict(), 'mlp.params')

# 加载模型
clone=MLP()# 随机初始化一下【类似new一个对象,之后再传参】
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
# 关闭dropout等随机性，使结果更固定、更稳定
# 训练时的dropout可以防止weight过拟合，之后经过dropout的weight再关闭dropout进行eval()

y_clone=clone(X)
# Y是由net产生的，y_clone已进入eval()模式，现在评估Y也就是评估net的模式
print(y_clone==Y)
# 返回是True,是因为MLP里面没施加随机因素,但net其实仍处在训练状态
# 如果有dropout，大概率输出是False