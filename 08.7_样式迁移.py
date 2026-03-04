import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
content_img=d2l.Image.open('./img/content.png')
d2l.plt.imshow(content_img)
d2l.plt.show()

style_img=d2l.Image.open('./img/autumn-oak.jpg')
d2l.plt.imshow(style_img)
d2l.plt.show()

rgb_mean=torch.tensor([0.485, 0.456, 0.406])
rgb_std=torch.tensor([0.229, 0.224, 0.225])

# 预处理和后处理
# PyTorch 使用通道优先，而图像库使用通道最后
def preprocess(img,image_shape):
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(rgb_mean,rgb_std)
    ])

    return transform(img).unsqueeze(0)

def postprocess(img):
    img=img[0].to(rgb_std.device)
    # 反标准化，恢复原始像素强度
    img=torch.clamp(
        img.permute(1,2,0)*rgb_std+rgb_mean,0,1
    )
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# 抽取图像特征
pretrained_net=torchvision.models.vgg19(pretrained=True)
style_layers,content_layers=[0,5,10,19,28],[25]

net=nn.Sequential(*[
    # pretrained_net.features：只取 VGG19 的特征提取部分（即卷积层和池化层）
    # 丢弃最后的分类层（全连接层）
    # .classifier：最后几层全连接层
    pretrained_net.features[i]
    for i in range(max(content_layers+style_layers)+1)
]
)

# 关键层特征图抽取
def extract_features(X,content_layers,style_layers):
    contents=[]
    styles=[]
    for i in range(len(net)):
        X=net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents,styles

def get_contents(image_shape,device):
    content_X=preprocess(content_img,image_shape).to(device)
    contents_Y,_=extract_features(content_X,content_layers,style_layers)
    return content_X,contents_Y

def get_styles(image_shape,device):
    style_X=preprocess(style_img,image_shape).to(device)
    _,styles_Y=extract_features(style_X,content_layers,style_layers)
    return style_X,styles_Y

# 定义损失函数
def content_loss(Y_hat,Y):
    # # 计算合成图特征 Y_hat 与原图特征 Y 之间的均方误差 (MSE)
    return torch.square(Y_hat-Y.detach()).mean()

def gram(X):
    # 格拉姆矩阵:计算颜色/纹理之间的相关性
    num_channels,n=X.shape[1],X.numel()//X.shape[1]
    X=X.reshape((num_channels,n))
    return torch.matmul(X,X.t()) / (num_channels*n)

def style_loss(Y_hat,gram_Y):
    # 比较风格
    return torch.square(gram(Y_hat)-gram_Y.detach()).mean()

def tv_loss(Y_hat):
    # 降噪（使平滑）
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

content_weight,style_weight,tv_weight=1,1e4,10
# 风格转移的损失是内容损失、风格损失和总变化损失的加权和
def compute_loss(X,contents_Y_hat,styles_Y_hat,contents_Y,styles_Y):
    contents_l=[
        content_loss(Y_hat,Y)*content_weight
        for Y_hat,Y in zip(contents_Y_hat,contents_Y)
    ]

    styles_l=[
        style_loss(Y_hat,Y)*style_weight
        for Y_hat,Y in zip(styles_Y_hat,styles_Y)
    ]

    tv_l=tv_loss(X)*tv_weight
    # 列表 × 10:这是“列表复制”
    # 它会把列表里的元素重复 10 遍，变成一个长度为 50 的长列表
    l=sum(styles_l+contents_l+[tv_l])
    return contents_l,styles_l,tv_l,l

class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X

device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)

d2l.plt.imshow(postprocess(output))
d2l.plt.show()