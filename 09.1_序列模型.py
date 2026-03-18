import torch
from torch import nn
from d2l import torch as d2l

# 生成数据
T=1000
time=torch.arange(1,T+1,dtype=torch.float32)
x=torch.sin(0.01*time)+torch.normal(0,0.2,(T,))
d2l.plot(time,[x],'time','x',xlim=[1,1000],figsize=(6,3))
d2l.plt.show()


tau=4
# 初始化特征矩阵，形状为 (996, 4)
features=torch.zeros((T-tau,tau))
# 第 0 列：x[0], x[1], ..., x[995]
# 第 1 列：x[1], x[2], ..., x[996]
# ...
for i in range(tau):
    features[:,i]=x[i:T-tau+i]
# 第5列（滑动窗口）
labels=x[tau:].reshape((-1,1))

batch_size=16
n_train=600
train_iter=d2l.load_array((features[:n_train],labels[:n_train]),batch_size,is_train=True)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net=nn.Sequential(nn.Linear(4,10),nn.ReLU(),nn.Linear(10,1))
    net.apply(init_weights)
    return net

loss=nn.MSELoss()

def train(net,train_iter,loss,epochs,lr,):
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    for epoch in range(epochs):
        for X,y in train_iter:
            trainer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            trainer.step()
        print(f'epoch{epoch+1},'
              f'loss:{d2l.evaluate_loss(net,train_iter,loss)}')

net=get_net()
epoch=5
lr=0.01
train(net,train_iter,loss,epoch,lr)

# epoch1,loss:0.1301351881733066
# epoch2,loss:0.06796270403030671
# epoch3,loss:0.060733341170768985
# epoch4,loss:0.060498087764962724
# epoch5,loss:0.058324809725347315

# 单步预测
# 用（996，4）预测第5列
onestep_preds=net(features)
d2l.plot(
    [time,time[tau:]],
    [x.detach().numpy(),onestep_preds.detach().numpy()],
    'time','x',
    legend=['data','l-step preds'],
    figsize=(6,3)
)
d2l.plt.show()

# 多步预测（持续预测）
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()
# 前600参与训练


# 多步预测（非持续）
max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.show()