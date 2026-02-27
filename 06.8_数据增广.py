import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# d2l.set_figsize()
# img=d2l.Image.open('./img/cat1.jpg')
# img=img.resize((224,224))
# d2l.plt.imshow(img)
# d2l.plt.show()
#
# # aug 本质上是一个函数或对象
# def apply(img,aug,num_rows=2,num_cols=4,scale=1.5):
#     Y=[aug(img) for _ in range(num_rows*num_cols)]
#     d2l.show_images(Y,num_rows,num_cols,scale=scale)

# # 左右翻转
# apply(img, torchvision.transforms.RandomHorizontalFlip())
# d2l.plt.show()

# # 上下翻转
# apply(img, torchvision.transforms.RandomVerticalFlip())
# d2l.plt.show()

# # 随即剪裁并放缩
# # scale=(0.1, 1)：面积比例。每次随机抠图时，抠出来的面积占原图总面积的百分比
# # ratio=(0.5, 2.0)：宽高比。控制裁剪框是扁的还是长的。$0.5$ 是瘦高型，$2.0$ 是矮胖型。（长/宽）
# shape_aug=torchvision.transforms.RandomResizedCrop((224,224),scale=(0.1,1),ratio=(0.5,2.0))
# apply(img,shape_aug)
# d2l.plt.show()

# #随机改变图像亮度
# # contrast=0：对比度不
# # saturation=0：饱和度
# # hue=0：色调
# apply(img,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,hue=0,saturation=0))
# d2l.plt.show()

# # 随机改变上述参数
# apply(img,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5,saturation=0.5))
# d2l.plt.show()

# # 叠加多种增广方式
# color_aug=torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5,saturation=0.5)
# shape_aug=torchvision.transforms.RandomResizedCrop((224,224),scale=(0.1,1),ratio=(0.5,2.0))
# aug=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),color_aug,shape_aug])
#
# apply(img,aug)
# d2l.plt.show()

# 使用图像增广进行训练
all_images=torchvision.datasets.CIFAR10(train=True,root='./data',download=True)
# range(32) 决定了我们取前 32 张图。
d2l.show_images([all_images[i][0] for i in range(32)],4,8,scale=0.8)
d2l.plt.show()

# 水平翻转
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor() # 变4D
])

# 不操作
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def load_cifar10(augs, batch_size, is_train=True):
    dataset=torchvision.datasets.CIFAR10(root='./data',train=is_train,transform=augs,download=True)
    dataloader=torch.utils.data.DataLoader(
        dataset,batch_size=batch_size,shuffle=is_train,num_workers=4
    )
    return dataloader


def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多 GPU 进行小批量训练"""
    if isinstance(X, list):
        # 如果 X 是列表（通常在某些多输入模型中），将每一项都复制到 GPU
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])

    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()

    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()  # 反向传播

    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)

    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用图像增广和多 GPU 训练模型"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])

    # 使用 DataParallel 在多个 GPU 上分发模型
    net = torch.nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in range(num_epochs):
        # 4个指标：总损失，总准确度，样本数，特征数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()

            # 每训练 1/5 的批次，更新一次动画图表
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3], None))

        # 每个 epoch 结束，在测试集上评估
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

batch_size=256
devices=d2l.try_all_gpus()
net=d2l.resnet18(10)
net[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

def init_weights(m):
    # 对全连接层和卷积层进行 Xavier 均匀分布初始化
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

# 将初始化应用到网络模型
net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    # 调用你之前写的 load_cifar10 函数加载数据
    # 注意：这里的参数顺序应与你在代码中定义的 load_cifar10(augs, batch_size, is_train) 一致
    train_iter = load_cifar10(train_augs, batch_size, True)
    test_iter = load_cifar10(test_augs, batch_size, False)

    # 定义损失函数：交叉熵损失，reduction="none" 是为了方便多 GPU 累加计算
    loss = nn.CrossEntropyLoss(reduction="none")

    # 定义优化器：使用 Adam 优化器通常比 SGD 收敛更快
    trainer = torch.optim.Adam(net.parameters(), lr=lr)

    # 启动多 GPU 训练循环
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

# 训练
train_with_data_aug(train_augs, test_augs, net)
d2l.plt.show()
# loss 0.262, train acc 0.909, test acc 0.772
# 6376.3 examples/sec on [device(type='cuda', index=0)]
train_with_data_aug(test_augs, test_augs, net)
d2l.plt.show()
# loss 0.048, train acc 0.984, test acc 0.799
# 6453.7 examples/sec on [device(type='cuda', index=0)]