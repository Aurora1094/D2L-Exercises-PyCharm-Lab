#加载图像分类数据集
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

trans=transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
print(len(mnist_train),len(mnist_test))

X, y = mnist_train[0]  #feature、label
print(X.shape, X.dtype, X.min().item(), X.max().item(), y)

#可视化数据集
import matplotlib.pyplot as plt

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if torch.is_tensor(img):
            ax.imshow(img.numpy(), cmap='gray')
        else:
            ax.imshow(img)

        if titles:
            ax.set_title(titles[i])
    return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
# d2l.plt.show()
def main():
    batch_size=256
    def get_dataloader_workers():
        return 4
    train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())
    timer=d2l.Timer()
    for X,y in train_iter:
        continue
    print(f'{timer.stop():.2f}sec')

if __name__ == "__main__":

    main()