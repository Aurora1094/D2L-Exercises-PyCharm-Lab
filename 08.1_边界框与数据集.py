import torch
from d2l import torch as d2l

d2l.set_figsize()
img=d2l.plt.imread('./img/catdog.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

# 角落坐标（左上马，右下），中心坐标（中心，长宽）
# 角落坐标->中心坐标
def box_corner_to_center(boxes):
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    cx=(x1+x2)/2
    cy=(y1+y2)/2
    w=x2-x1
    h=y2-y1
    boxes=torch.stack((cx,cy,w,h),axis=-1)
    return boxes

def box_center_to_center(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

dog_bbox=[12.0,55.0,110.0,135.0]
cat_bbox=[85.0,10.0,140.0,125.0]

boxes=torch.tensor([dog_bbox,cat_bbox])
print(box_center_to_center(box_corner_to_center(boxes))==boxes)

def bbox_to_rect(bbox,color):
    return d2l.plt.Rectangle(xy=(bbox[0],bbox[1]),
                             width=bbox[2]-bbox[0],
                             height=bbox[3]-bbox[1],
                             fill=False,
                             edgecolor=color,
                             linewidth=2)

fig=d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox,'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox,'red'))
d2l.plt.show()

import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

# 将香蕉检测数据集的下载链接和对应的 SHA-1 哈希值注册到 d2l 的数据仓库中
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72'
)

# AI
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    # 下载并解压数据集，返回本地路径
    data_dir = d2l.download_extract('banana-detection')

    # 根据训练集或测试集选择对应的 CSV 文件
    csv_fname = os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'label.csv')

    # 使用 pandas 读取 CSV，并将图片名称设为索引
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')

    images, targets = [], []

    # 遍历 CSV 中的每一行
    for img_name, target in csv_data.iterrows():
        # 读取图像文件并转换为 PyTorch 张量
        images.append(torchvision.io.read_image(
            os.path.join(data_dir,
                         'bananas_train' if is_train else 'bananas_val',
                         'images', f'{img_name}')))

        # target 包含：类别(class) 和 4个边界框坐标
        targets.append(list(target))

    # 返回图像列表和标签张量
    # unsqueeze(1) 是为了给标签增加一个维度，通常用于后续的 Batch 处理
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features,self.labels=read_data_bananas(is_train)
        print('read'+str(len(self.features))+(f'training examples' if is_train else f'validation data'))
    def __getitem__(self, idx):
        return (self.features[idx].float(),self.labels[idx])
    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    train_iter=torch.utils.data.DataLoader(BananasDataset(is_train=True),batch_size=batch_size,shuffle=True)
    val_iter=torch.utils.data.DataLoader(BananasDataset(is_train=False),batch_size=batch_size)
    return train_iter, val_iter

batch_size=32
edge_size=256
train_iter, val_iter=load_data_bananas(batch_size=batch_size)
batch=next(iter(train_iter))
print(batch[0].shape,batch[1].shape)

imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255

# 2. 创建画布并显示图片
# 2, 5 代表画 2 行 5 列，共 10 张图
axes = d2l.show_images(imgs, 2, 5, scale=2)

# 3. 在图片上绘制边界框
for ax, label in zip(axes, batch[1][0:10]):
    # label[0] 是当前图片的所有目标标签
    # [1:5] 取出坐标部分 (x1, y1, x2, y2)
    # * edge_size 是将归一化的坐标还原回像素坐标（因为图片是 256x256，通常 edge_size=256）
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

d2l.plt.show()