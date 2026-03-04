import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

pretrained_net = torchvision.models.resnet18(pretrained=True)
# print(list(pretrained_net.parameters()))
# print(list(pretrained_net.children())[0:-3])

# 去掉后两层
net=nn.Sequential(*list(pretrained_net.children())[:-2])

X=torch.randn(1,3,320,480)
print(net(X).shape)

num_classes=21
net.add_module('final_conv',nn.Conv2d(512,num_classes,kernel_size=1))
# 转置卷积中，stride其实也就是放大倍数
net.add_module('transpose_conv',
               nn.ConvTranspose2d(
                   num_classes,
                   num_classes,
                   kernel_size=64,
                   padding=16,
                   stride=32
               ))

# 转置卷积的初始化（双线性插值）
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));

img = torchvision.transforms.ToTensor()(d2l.Image.open('./img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()

d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img)
d2l.plt.show()

W=bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)
batch_size=32
crop_size=(320,480)
train_iter,test_iter=d2l.load_data_voc(batch_size,crop_size)

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
    # return F.binary_cross_entropy(inputs, targets, reduction='none')
num_epochs=5
lr=0.001
wd=1e-3
devices=d2l.try_all_gpus()
trainer=torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss,trainer, num_epochs=num_epochs,devices=devices)
d2l.plt.show()

# loss 0.412, train acc 0.871, test acc 0.852
# 112.6 examples/sec on [device(type='cuda', index=0)]

def predict(img):
    X=test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred=net(X.to(devices[0])).argmax(1)
    return pred.reshape(pred.shape[1],pred.shape[2])

def label2image(pred):
    # d2l.VOC_COLORMAP：这是一个预定义的列表
    colormap=torch.tensor(d2l.VOC_COLORMAP,device=devices[0])
    X=pred.long()
    # 输入 X: 形状是 (H, W)
    # 返回结果 colormap[X, :]: 形状变成了 (H, W, 3)
    # 3是RGB通道
    # 遍历 X 中的每一个数字，去 colormap 里找到对应的 RGB 颜色
    return colormap[X,:]

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
d2l.plt.show()