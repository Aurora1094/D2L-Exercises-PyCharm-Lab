import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 下载热狗数据集
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                             'fba480ffa8aa7e0febb511d181409f899b9baa53')
data_dir = d2l.download_extract('hotdog')
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs=[train_imgs[i][0] for i in range(8)]
# 当 i=0 时，索引是 -1（最后一张图）
not_hotdogs=[train_imgs[-i-1][0] for i in range(8)]
d2l.show_images(hotdogs+not_hotdogs,2,8,scale=1.4)
d2l.plt.show()

# 数据增广
# RGB上的mean和std（ImageNet特定数值）
normalize=torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])

train_augs=torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

# Resize(256): 先把图片统一缩放到 $256 \times 256$。
# CenterCrop(224): 从正中心裁剪出 $224 \times 224$ 的区域。
test_augs=torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

# 定义和初始化模型
pretrained_net = torchvision.models.resnet18(pretrained=True)
# fc 代表 Fully Connected（全连接层）
print(pretrained_net.fc)

# 修改模型
finetune_net = torchvision.models.resnet18(pretrained=True)
# 替换输出层,因为热狗数据集只输出yes or no
# 拆掉了模型原有的输出层，换上了一个完全随机、没有任何知识的新层
finetune_net.fc = nn.Linear(in_features=finetune_net.fc.in_features, out_features=2)
# 新零件（新的 fc 层）：默认情况下，PyTorch 会用一种通用的随机方式初始化它
# Xavier 初始化：让每一层输出的方差等于输入的方差
nn.init.xavier_uniform_(finetune_net.fc.weight)

def train_fine_tuning(net,learning_rate,batch_size=128,num_epochs=0,param_group=True):
    train_iter=torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'),transform=train_augs),
        batch_size=batch_size,shuffle=True
    )

    test_iter=torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'),transform=test_augs),
        batch_size=batch_size,shuffle=False
    )

    devices=d2l.try_all_gpus()

    loss=nn.CrossEntropyLoss(reduction='none')

    if param_group:
        # 把模型中除了最后那层 fc 之外的所有预训练参数（即卷积层）都挑出来
        param_1x=[
            param for name,param in net.named_parameters()
            if name not in ["fc.weight", "fc.bias"]
        ]

        # fc 层是你刚刚换上去的,所以学习率要大一点
        trainer=torch.optim.SGD(
            [ # 针对其他和fc
                {'params': param_1x},{'params':net.fc.parameters(),'lr':learning_rate*10}
            ],lr=learning_rate,weight_decay=0.001
        )
    else:
        trainer=torch.optim.SGD(net.parameters(),lr=learning_rate,weight_decay=0.001)

    d2l.train_ch13(net,train_iter,test_iter,loss,trainer,10,devices)

train_fine_tuning(finetune_net,5e-5,256,5)
d2l.plt.show()

# loss 0.143, train acc 0.950, test acc 0.939
# 290.4 examples/sec on [device(type='cuda', index=0)]

scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(in_features=scratch_net.fc.in_features, out_features=2)
train_fine_tuning(scratch_net,5e-5,256,5)
d2l.plt.show()

# loss 0.351, train acc 0.852, test acc 0.861
# 293.6 examples/sec on [device(type='cuda', index=0)]