# 多尺度锚框
import torch
from d2l import torch as d2l

img=d2l.plt.imread('./img/catdog.jpg')
h,w=img.shape[:2]
print(h,w)

# 在特征图上生成锚框，每个单位作为锚框中心
def display_anchors(fmap_w,fmap_h,s):
    d2l.set_figsize()
    # 批量大小 1 和通道数 10 是随机设置的
    fmap=torch.zeros((1,10,fmap_h,fmap_w))
    # 为fmap每个像素点生成m+n-1锚框
    # (同时归一化【有利于后续还原真实坐标】)
    anchors=d2l.multibox_prior(
        fmap,sizes=s,ratios=[1,2,0.5]
    )
    # 用于还原真实坐标
    bbox_scale=torch.tensor((w,h,w,h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,anchors[0]*bbox_scale)

display_anchors(2,2,[0.15,0.3])
d2l.plt.show()

# 单发多框检测（SSD）
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 每个像素点的不同检测物的预测值输出
def cls_predictor(num_inputs,num_anchors,num_classes):
    return nn.Conv2d(num_inputs,num_anchors*(num_classes+1),kernel_size=3,padding=1)

# 每个像素点的中心坐标偏移输出
def bbox_predictor(num_inputs,num_anchors):
    return nn.Conv2d(num_inputs,num_anchors*4,kernel_size=3,padding=1)

# 连接多尺度的预测
def forward(x,block):
    return block(x)
# 哦哦哦！这个是定义好的！！：(Batch, Channel, Height, Width)
Y1=forward(torch.zeros((2,8,20,20)),cls_predictor(num_inputs=8,num_anchors=5,num_classes=10))
Y2=forward(torch.zeros((2,16,10,10)),cls_predictor(num_inputs=16,num_anchors=3,num_classes=10))

# 4D->2D
def flatten_pred(pred):
    # permute(0, 2, 3, 1)：把维度变成 (Batch, H, W, Channel)
    # torch.flatten(..., start_dim=1)：保留 Batch 维度， H, W, C 相乘
    # 每一行是一张图片
    return torch.flatten(pred.permute(0,2,3,1),start_dim=1)

def concat_preds(preds):
    # 将图像拼在一起
    return torch.cat([flatten_pred(p) for p in preds],dim=1)

# print(concat_preds([Y1,Y2]).shape)

# 下采样CNN(高宽减半)
def down_sample_blk(in_channels,out_channels):
    blks=[]
    for _ in range(2):
        blks.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        blks.append(nn.BatchNorm2d(out_channels))
        blks.append(nn.ReLU())
        in_channels=out_channels
    blks.append(nn.MaxPool2d(2))
    return nn.Sequential(*blks)

# 基本网络块（输入原始图片到fmap）
def base_net():
    blks=[]
    num_fliter=[3,16,32,64]
    for i in range(len(num_fliter)-1):
        blks.append(down_sample_blk(num_fliter[i],num_fliter[i+1]))
    return nn.Sequential(*blks)

Y=forward(torch.zeros((2,3,256,256)),base_net())
print(Y.shape)

# SSD由5个模块组成:是为了得到很多个fmap
def get_blk(i):
    if i==0:
        blk=base_net()
    elif i==1:
        blk=down_sample_blk(64,128)
    elif i==4:
        blk=nn.AdaptiveAvgPool2d(1)
    else:
        blk=down_sample_blk(128,128)
    return blk

# 前向传播
def blk_forward(x,blk,size,ratio,cls_predictor,bbox_predictor):
    Y=blk(x)
    # d2l.multibox_prior()生成一组锚框
    anchors=d2l.multibox_prior(Y,sizes=size,ratios=ratio)
    cls_pred=cls_predictor(Y)
    bbox_pred=bbox_predictor(Y)
    return (Y,anchors,cls_pred,bbox_pred)

# 超参数
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

net = TinySSD(num_classes=1)
X=torch.zeros((32,3,256,256))
anchors, cls_preds, bbox_preds = net(X)

# 香蕉数据集
batch_size=32
lr = 0.2
weight_decay = 5e-4
train_iter,_=d2l.load_data_bananas(batch_size)
device,net=d2l.try_gpu(),TinySSD(1)
trainer = torch.optim.SGD(net.parameters(), lr, weight_decay)

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

# 分类和位置预测
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # reshape(batch_size, -1)：把算出来的损失重新按图片分回组
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # bbox_masks抹除背景（背景是0）
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    # argmax(dim=-1) 在最后一个维度（类别维度）找最大值的下标作为分类预测
    predicted_classes = cls_preds.argmax(dim=-1)
    correct_predictions = (predicted_classes.type(cls_labels.dtype) == cls_labels)
    return float(correct_predictions.sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 训练
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量（和真实的Y进行对应）
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
d2l.plt.show()

# unsqueeze(0):刚读入的图片是 (3, H, W),变成 (1, 3, H, W)
X=torchvision.io.read_image('../data/banana.jpg').unsqueeze(0).float()
# permute(1, 2, 0): PyTorch 处理图片用的是 (通道, 高, 宽)，但 Matplotlib 绘图要求是 (高, 宽, 通道)
img=X.squeeze(0).permute(1,2,0).long()

# 针对单张图片
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    # softmax要求类别维度在中间，所以我们需要交换一下维度
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # multibox_detection
    # 坐标转换: 把“偏移量”应用到“锚框”上，计算出预测框在图上的真实 (x, y)。
    # 置信度过滤: 丢掉那些概率太低的“垃圾框”。
    # NMS (非极大值抑制): 如果好几个框都圈中了同一个香蕉，它只保留概率最高的那一个，把其余重复的删掉。
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    # d2l.multibox_detection 函数返回的 output 形状通常是 (批量大小, 锚框总数, 6)
    # 这 6 列的数据格式是：
    # [类别ID, 置信度, x1, y1, x2, y2]
    # multibox_detection 运行完后，那些被判定为“背景”或者被 NMS 删掉的框，其类别 ID 会被标记为 -1。
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    # 0是代表第一张图
    return output[0, idx]

# 处理一个batch
# outputs = predict(batch_X) # 假设返回形状为 (Batch, Anchors, 6)
# for i in range(batch_X.shape[0]):
#     # 针对每一张图进行过滤和显示
#     valid_idx = [j for j, row in enumerate(outputs[i]) if row[0] != -1]
#     display(batch_imgs[i], outputs[i, valid_idx], threshold=0.9)

output = predict(X)

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        # fig.axes: 告诉函数画在刚才那张原图的坐标系里
        # d2l.show_bboxes 画框
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')


display(img, output.cpu(), threshold=0.9)
d2l.plt.show()
