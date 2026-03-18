import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# print(dir(d2l))

batch_size=32
num_steps=35
data = d2l.TimeMachine(batch_size=batch_size, num_steps=num_steps)
train_iter = data.get_dataloader(train=True)
vocab = data.vocab

print(F.one_hot(torch.tensor([0,2]),len(vocab)))

# 模拟输入数据
X=torch.arange(10).reshape((2,5))
# 转置是为了后面循环逻辑每次把时间步上的单词按先后顺序参与遍历
print(F.one_hot(X.T,28).shape)

# 初始化参数
def get_params(vocab_size,num_hiddens,device):
    # 输入onehot字符向量，输出下一字符概率分布
    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape).to(device)*0.01

    W_xh=normal((num_inputs,num_hiddens))
    W_hh=normal((num_hiddens,num_hiddens))
    b_h=torch.zeros(num_hiddens,device=device)
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)
    params=[W_xh,W_hh,b_h,W_hq,b_q]

    for param in params:
        param.requires_grad=True

    return params

# 如果是普通 RNN，盒子里就装：(H,)
# 如果是 LSTM，盒子里就装：(H, C)
def init_rnn_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)

# 前向传播计算
def rnn(inputs,state,params):
    W_xh,W_hh,b_h,W_hq,b_q=params
    # 等同于 H = state[0]【元组第一个】
    H, =state
    outputs=[]
    # 次数是第一维
    for X in inputs:
        H=torch.tanh(torch.mm(X,W_xh) + torch.mm(H,W_hh)+b_h)
        Y=torch.mm(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs,0),(H, )

# 用类包装函数
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    # 为什么要用 __call__？
    # 果你把计算逻辑写在一个普通函数里，比如叫 forward
    # output, new_state = net.forward(X, state)  # 必须显式调用 .forward
    # 当定义了 def __call__(self, X, state)
    # Python 允许省去方法名，直接对实例对象加圆括号
    # output, new_state = net(X, state)
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# 检查输出形状
num_hiddens=512
net = RNNModelScratch(vocab_size=len(vocab), num_hiddens=num_hiddens,device=d2l.try_gpu(),get_params=get_params, init_state=init_rnn_state, forward_fn=rnn)
state=net.begin_state(X.shape[0],d2l.try_gpu())
Y,new_state=net(X.to(d2l.try_gpu()),state)
print(Y.shape)
print(new_state[0].shape)

def predict_ch8(prefix,num_preds,net,vocab,device):
    # state初始化，outputs输入第一个字符
    state=net.begin_state(batch_size=1,device=device)
    outputs=[vocab[prefix[0]]]
    get_input=lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # state晚于outputs输入一个
    for y in prefix[1:]:
        _,state=net(get_input(),state)
        outputs.append(vocab[y])
    # 输出
    for _ in range(num_preds):
        y,state=net(get_input(),state)
        # 记录猜测的输出
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

predict_ch8('time traveller ',10,net,vocab,d2l.try_gpu())

# 梯度裁剪
def grad_clipping(net,theta):
    # 检查是否为官方模型
    if isinstance(net,nn.Module):
        params=[p for p in net.parameters()if p.requires_grad]
    else:
        params=net.params
    norm=torch.sqrt(sum(
        torch.sum(p.grad**2)

    for p in params))

    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm

def train_epoch_ch8(net,train_iter,loss,updater,device,use_random_iter):
    state=None

    timer=d2l.Timer()
    metric=d2l.Accumulator(2)
    for X,Y in train_iter:
        bs=X.shape[0]
        # 第一个批次或随机抽样说明时间上不连续，需要将隐藏状态初始化为0
        if state is None or use_random_iter:
            state=net.begin_state(X.shape[0],device=device)
        else:
            # 用的是 PyTorch 官方的普通 RNN 或 GRU 模型
            if isinstance(state, tuple):
                last_bs = state[0].shape[0] if state[0].dim() == 2 else state[0].shape[1]
            else:
                last_bs = state.shape[1]

            if bs != last_bs:
                # 遇到数据集末尾的残缺批次，强制重新初始化状态，避免矩阵相加时报维度不匹配错误
                state = net.begin_state(bs, device=device)
                # ========================================================
            else:
                # 正常情况：用的是 PyTorch 官方的普通 RNN 或 GRU 模型
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # 为了防止在反向传播时计算图无限向后追溯
                    state.detach_()
                else:
                    for s in state:
                        s.detach_()
        # 在语言模型中，通常将时间步作为第一维度，批次作为第二维度
        y=Y.T.reshape(-1)
        X,y=X.to(device),y.to(device)
        y_hat,state=net(X,state)
        l=loss(y_hat,y.long()).mean()
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net,1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net,1)
            updater(batch_size=1)
        metric.add(l.item()*y.numel(),y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):

    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs= 500
lr=1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)

train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),use_random_iter=True)

d2l.plt.show()