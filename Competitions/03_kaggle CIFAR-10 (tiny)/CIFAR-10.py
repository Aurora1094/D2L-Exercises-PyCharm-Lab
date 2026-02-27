import collections
import math
import os
import shutil
import pandas as pd
import torchvision
from torch import nn
from d2l import torch as d2l
import torch
from matplotlib import pyplot as plt

d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b2975444')

demo = True

if demo:
    data_dir = r'C:\Users\19902\Desktop\data\kaggle_cifar10_tiny'
else:
    data_dir = '../data/cifar-10/'

# 整理数据集
def read_csv_labels(fname):
    # 只读模式
    with open(fname,'r') as f:
        lines=f.readlines()[1:]
    # rstrip() 去掉行尾的换行符
    # split(',') 按逗号分割，将每一行变成 ['文件名', '标签名'] 的列表
    # 原始：
    # lines = [
    #     "1,frog\n",
    #     "2,truck\n",
    #     "3,truck\n"
    # ]
    tokens=[l.rstrip().split(',') for l in lines]
    return dict(((name,label)for name,label in tokens))

labels=read_csv_labels(os.path.join(data_dir,'trainLabels.csv'))
print(labels)

def copyfile(filename,target_dir):
    os.makedirs(target_dir,exist_ok=True)
    shutil.copy(filename,target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    # 1. 计算训练集中样本最少的类别有多少张图片
    n = collections.Counter(labels.values()).most_common()[-1][1]

    # 2. 根据比例计算每一类应该分给验证集的图片数量
    n_valid_per_label = max(1, math.floor(n * valid_ratio))

    label_count = {}

    # 3. 遍历原始训练文件夹中的所有图片文件
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        # 获取文件名（不含后缀）对应的标签
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)

        # 4. 核心逻辑：分拣文件
        # 如果当前类别的计数还没达到验证集要求的数量，就拷贝到验证集目录
        copyfile(
            fname,
            os.path.join(data_dir, 'train_valid',
                         'test' if label not in label_count or label_count[label] < n_valid_per_label else 'train',
                         label)
        )

        # 更新该类别的计数
        label_count[label] = label_count.get(label, 0) + 1
    return n_valid_per_label

def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        # 从1路径到2路径copy
        # data/test/123.png -> data/train_valid_test/test/unkown/123.png
        copyfile(
        os.path.join(data_dir, 'test',test_file),
        os.path.join(data_dir, 'train_valid_test', 'test','unkown')
        )

def reorg_cifar10_data(data_dir,valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir,'trainLabels.csv'))
    # 划分搬运训练集和验证集
    reorg_train_valid(data_dir,labels,valid_ratio)
    # 搬运测试集图片
    reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
# 图片增广
transform_train=torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32,scale=(0.64,1.0),ratio=(1.0,1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])
])

# 读取数据集
# torchvision.datasets.ImageFolder会自动读取指定路径下的子文件夹
# 文件夹名会被当作“标签”（Label），里面的图片会被当作“样本”
# train_ds,train_valid_ds=[
#     torchvision.datasets.ImageFolder(
#         os.path.join(data_dir, 'train_valid_test',folder),
#         transform=transform_train)
#     # 现在还没增广，当用 DataLoader 开始迭代（如 for X, y in train_iter）时
#     # 程序每读到一张图，才会按照transform增广操作
#     for folder in ['train','train_valid_test']
# ]
#
# valid_ds,test_ds=[
#     torchvision.datasets.ImageFolder(
#         os.path.join(data_dir, 'train_valid_test',folder),
#         transform=transform_test)
#     for folder in ['valid', 'test']
# ]

# 1. 读取训练集和“训练+验证”集
# 注意：根据你的 reorg_train_valid 函数，父目录是 'train_valid'
train_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid', 'train'),
    transform=transform_train)



# 2. 读取验证集和测试集
# 验证集：你在 reorg_train_valid 里把验证集文件夹命名为了 'test'
valid_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid', 'test'),
    transform=transform_test)

# 全量训练集
train_valid_ds = torch.utils.data.ConcatDataset([train_ds, valid_ds])

# 测试集：你在 reorg_test 里把测试集放在了 'train_valid_test' 目录下
test_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', 'test'),
    transform=transform_test)

train_iter,train_valid_iter=[
    torch.utils.data.DataLoader(
        dataset,batch_size,shuffle=True,drop_last=True
    )for dataset in [train_ds,train_valid_ds]
]

valid_iter,test_iter=[
    torch.utils.data.DataLoader(
    ds,batch_size,shuffle=False,drop_last=False
    )for ds in [valid_ds,test_ds]
]

# 模型
def get_net():
    num_classes=10
    net=d2l.resnet18(num_classes,3)
    return net

# 损失函数
loss=nn.CrossEntropyLoss(reduction='none')

# AI
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    # 1. 定义优化器：使用带动量的 SGD，并配置权重衰减（L2 正则化）
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    # 2. 定义学习率调度器：每隔 lr_period 个 epoch，将学习率乘以 lr_decay
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    # 3. 初始化可视化工具和计时器
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)

    # 4. 启用多 GPU 数据并行训练
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    # 5. 主训练循环
    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式
        metric = d2l.Accumulator(3)  # 累加器：存储训练损失之和、训练准确度之和、样本数

        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()

            # 每完成 1/5 的批次，更新一次动画图表
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2], None))

        # 6. 如果有验证集，每个 epoch 结束后计算验证集准确率
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))

        # 7. 更新学习率
        scheduler.step()

    # 8. 打印最终统计结果
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')

# def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
num_epochs = 100
lr = 2e-4
wd=5e-4
lr_period = 4 #每隔4个epoch，lr=lr*lr_decay
lr_decay = 0.9

# train_ds	base\train_valid\train
# valid_ds	base\train_valid\test
# train_valid_ds	base\train_valid
# test_ds	base\train_valid_test\test

# 第一次：用train_ds训练，valid_ds验证，计算acc辅助调参
# 第二次：用train_valid_ds训练
# 第三次：预测test_ds

net=get_net()
# 第一次
train(net,train_iter,valid_iter,num_epochs,lr,wd,d2l.try_all_gpus(), lr_period, lr_decay)
plt.show()

net=get_net()
# 第二次：全量训练
train(net, train_valid_iter, None, num_epochs, lr, wd, d2l.try_all_gpus(), lr_period, lr_decay)
plt.show()

# 第三次
preds =[]

# 切换到预测模式
net.eval()

for X, _ in test_iter:
    # 将数据移至 GPU 并进行前向传播
    y_hat = net(X.to(d2l.try_gpu()))
    # 获取概率最大的类别索引，转为 numpy 并存入 preds 列表
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

# 格式化结果并导出 CSV
# 创建 ID 列
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))

# 构造 DataFrame
df = pd.DataFrame({'id': sorted_ids, 'label': preds})

# 将数字索引转换回文字标签
# train_valid_ds.classes 是 ImageFolder 自动生成的类别列表
df['label'] = df['label'].apply(lambda x: train_ds.classes[x])
# 保存为 submission.csv
df.to_csv('submission.csv', index=False)

print("预测完成，结果已保存至 submission.csv")