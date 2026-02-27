import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import os
import matplotlib.pyplot as plt

# 下载数据集

# 1. 手动定义缺失的 URL 和 DATA_HUB
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB = dict()

# 2. 注册数据集信息
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# 3. 定义下载函数 (D2L 内部通常有这个，但脚本需要明确引用)
def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个 DATA_HUB 中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不在 {DATA_HUB} 中"
    url, sha1_hash = DATA_HUB[name]
    return d2l.download(url, cache_dir, sha1_hash=sha1_hash)

# 4. 执行下载并加载
train_data = pd.read_csv(download('kaggle_house_train'))
# test_data = pd.read_csv(download('kaggle_house_test'))
try:
    test_data = pd.read_csv('test.csv') # 读取你上传的本地文件
    print("✅ 成功读取本地 test.csv")
except FileNotFoundError:
    print("⚠️ 未找到本地 test.csv，尝试使用在线下载（可能会导致行数错误）...")
    test_data = pd.read_csv(download('kaggle_house_test'))

# 5. 打印结果验证
print(f"训练集维度: {train_data.shape}")
print(f"测试集维度: {test_data.shape}")

# 预览前 4 行数据和部分特征
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

#(一)处理无关特征
# 删除第一列特征ID，不应该学习，每个样本都不一样！
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


#（二）标准化
# 将缺失值替换为相应特征均值，之后标准化（每个特征放缩到均值为0，方差为1）
# MSZoning          object  <-- 这是一个字符串类别
numeric_features=all_features.dtypes[all_features.dtypes != 'object'].index
# Z-Score 标准化
# 语法：lambda 参数 : 表达式
# add_lambda = lambda a, b : a + b
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x:(x-x.mean())/(x.std())
)
# 标准化后，该列的均值变成了 0。因此，用 0 填充缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

#（三）处理离散值
# get_dummies()将特征转onehot编码
# 从生成的 $N$ 个 One-Hot 列中，删掉第一列，只保留剩下的 $N-1$ 列【删除冗余信息】
all_features=pd.get_dummies(all_features,drop_first=True)

#补丁：
all_features = all_features.apply(pd.to_numeric, errors='coerce')
all_features = all_features.fillna(0)
# (2919, 244)
# all_features=pd.get_dummies(all_features)
# (2919, 287)
# 你有 43 个类别特征，每个特征都被删掉了一列，正好就是 $287 - 43 = 244$ 列。
print(all_features.shape)

# 从 Pandas 数据框 (DataFrame) 到 PyTorch 张量 (Tensor)的转化【因为pandas更适用于数据处理】
# 无论是 Pandas 的 DataFrame，还是 PyTorch 的 Tensor，它们的核心都是线性代数里的矩阵（Matrix）或张量（Tensor）
n_train=train_data.shape[0]
train_features = torch.tensor(all_features.iloc[:n_train].values.astype(float), dtype=torch.float32)
test_features = torch.tensor(all_features.iloc[n_train:].values.astype(float), dtype=torch.float32)
train_labels=torch.tensor(train_data.SalePrice.values.reshape((-1,1)),dtype=torch.float32)

#训练
loss=nn.MSELoss()
in_features=train_features.shape[1]

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 64), # 第一层：将特征压缩到 64 维
        nn.ReLU(),
        # nn.Dropout(0),
        nn.Linear(64, 1)            # 输出层：将 64 维压缩到 1 维（预测的房价）
    )
    return net

# 估计相对误差（对数）
def log_rmse(net,features,labels):
    # 第一行：预测并清理数值
    # net(features) 是模型给出的房价预测值。
    # torch.clamp(..., 1, float('inf')) 把所有小于 1 的预测值强制变成 1。
    clipped_preds=torch.clamp(net(features),1,float('inf'))
    # 第二行：计算对数空间的均方根误差 (RMSE)
    rmse=torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    # 第三行：返回数值
    # .item() 把只有一个元素的张量转成 Python 的浮点数。
    # rmse是tensor运算得到的，虽然是一个数，但仍然是给tensor
    return rmse.item()

# k折交叉验证的切割
def get_k_fold_data(k, i, X, y):
    # 确保 k 大于 1，否则无法进行交叉验证
    assert k > 1

    # 计算每一折（每个数据块）的大小
    # // 是整数除法，确保得到的是整数行数
    fold_size = X.shape[0] // k

    X_train, y_train = None, None
    for j in range(k):
        # 定义当前这块数据的索引范围
        # slice(start, stop) 会生成一个切片对象
        idx = slice(j * fold_size, (j + 1) * fold_size)

        # 根据索引取出当前这一小块特征和标签
        X_part, y_part = X[idx, :], y[idx]

        # 如果当前的循环索引 j 等于我们指定的验证折索引 i
        if j == i:
            # 这一块作为本轮的“考卷”（验证集）
            X_valid, y_valid = X_part, y_part

        # 如果训练集还是空的（处理第一块非验证数据的情况）
        elif X_train is None:
            X_train, y_train = X_part, y_part

        # 剩下的情况：把当前块拼接到已有的训练集后面
        else:
            # torch.cat 用于拼接张量，0 代表按行拼接
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)

    return X_train, y_train, X_valid, y_valid

# 手写训练函数
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 使用 Adam 优化器，处理房价预测这种多特征问题效果很好
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 建立数据迭代器
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)  # 确保你前面定义了 loss = nn.MSELoss()
            l.backward()
            optimizer.step()

        # 每一轮记录对数误差
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls





# k折交叉验证逻辑
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        # 1. 调用之前定义的函数，获取第 i 折的训练集和验证集
        data = get_k_fold_data(k, i, X_train, y_train)

        # 2. 重新初始化网络（确保每一折都是从头开始练，不带之前的记忆）
        net = get_net()

        # 3. 这里的 *data 是 Python 的解包语法，会自动把返回的 4 个 tensor 传给 train
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)

        # 4. 累加最后一轮（epoch）的 log_rmse 误差
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        # 5. 画图（通常只画第一折的曲线来观察收敛情况，避免屏幕太乱）
        if i == 0:
            # d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
            #          xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
            #          legend=['train', 'valid'], yscale='log')
            plt.figure(figsize=(5, 3))
            plt.plot(range(1, num_epochs + 1), train_ls, label='train')
            plt.plot(range(1, num_epochs + 1), valid_ls, label='valid')
            plt.xlabel('epoch')
            plt.ylabel('rmse')
            plt.yscale('log')
            plt.legend()
            plt.show()

        # 6. 打印当前这一折的结果
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')

    # 最后返回 k 次实验的平均误差
    return train_l_sum / k, valid_l_sum / k

k=8
num_epochs=100
lr=0.08
weight_decay=0.2
batch_size=64

train_l,valid_l=k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k} 折验证，平均训练log rmse {float(train_l):f}, '
      f'平均验证log rmse {float(valid_l):f}')




def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    # 1. 再次确认你的 test_data 到底长什么样
    # 如果这里打印出来的行数不是 50，说明你一开始读取文件的时候就读错了！
    print(f"输入测试集行数: {len(test_data)}")

    net = get_net()
    # 训练
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')

    # 预测
    preds = net(test_features).detach().numpy()

    # ================= 修正：基于原始 ID 进行赋值 =================
    # 我们直接把预测值塞回原始的 test_data DataFrame 中
    # 这样能保证 ID 和 预测值 是一一对应的，不会错位
    test_data['value'] = pd.Series(preds.reshape(1, -1)[0])

    # 构造提交文件
    submission = pd.DataFrame()

    # 处理 ID 格式：将数字 ID 转换为 id_1, id_2 格式（如果尚未转换）
    # 假设 test_data 里有一列叫 'Id' (数字)
    if 'Id' in test_data.columns:
        submission['ID'] = 'id_' + test_data['Id'].astype(str)
    elif 'ID' in test_data.columns:
        # 如果已经是 ID 列，检查是否需要加前缀
        if pd.api.types.is_numeric_dtype(test_data['ID']):
            submission['ID'] = 'id_' + test_data['ID'].astype(str)
        else:
            submission['ID'] = test_data['ID']
    else:
        # 如果没有 ID 列，根据行数生成 (这是最后的保底，前提是顺序没乱)
        submission['ID'] = [f'id_{i}' for i in range(1, len(test_data) + 1)]

    # 把刚才对齐好的 value 放进去
    submission['value'] = test_data['value']

    # 最终检查：必须是 50 行
    if len(submission) != 50:
        print(f"❌ 错误：生成的行数是 {len(submission)}，但比赛要求 50 行！")
        print("请检查你是否读取了错误的 csv 文件？比如读成了训练集？")
    else:
        submission.to_csv('submission.csv', index=False)
        print("✅ 成功！submission.csv 已生成 (50行，ID已对齐)")
        print(submission.head())


# 运行
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)