import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np

image_size = [1, 28, 28]
latent_dim = 96
batch_size = 264
use_gpu = torch.cuda.is_available()

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),

            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),
            #  nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # shape of z: [batchsize, latent_dim]

        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)

        return image


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size, dtype=np.int32), 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        # shape of image: [batchsize, 1, 28, 28]

        prob = self.model(image.reshape(image.shape[0], -1))

        return prob


#MINIST
dataset = torchvision.datasets.MNIST(
    root = 'G:/1code/MINIST',  #数据集的位置
    train = True,       #如果为True则为训练集，如果为False则为测试集

    transform = torchvision.transforms.Compose([
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                             #  torchvision.transforms.Normalize([0.5], [0.5]),
                                         ]),


    download=True
)

dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size, drop_last=True)

# 实例化网络
generator = Generator()
discriminator = Discriminator()
# 定义优化器
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, )
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003)
# 定义损失函数
loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)

if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

# 训练
num_epoch = 300
for epoch in range(num_epoch):
    for i ,min_batch in enumerate(dataloader):
        # 真实数据
        gt_images, _ = min_batch
        # 虚拟数据
        z = torch.randn(batch_size, latent_dim)

        if use_gpu:
            gt_images = gt_images.to("cuda")
            z = z.to("cuda")

        #对生成器参数进行梯度下降
        pred_images = generator(z)

        g_optimizer.zero_grad()
        # l1 loss
        recons_loss = torch.abs(pred_images - gt_images).mean()
        g_loss = g_loss = recons_loss*0.05 + loss_fn(discriminator(pred_images), labels_one)
        g_loss.backward()
        g_optimizer.step()

        #对判别器参数进行梯度下降
        d_optimizer.zero_grad()
        real_loss = loss_fn(discriminator(gt_images), labels_one)
        fake_loss = loss_fn(discriminator(pred_images.detach()), labels_zero)
        d_loss = (real_loss + fake_loss)
        # d_loss = 0.5*loss_fn(discriminator(gt_image),torch.ones(batch_size,1) + loss_fn(discriminator(pred_images,torch.ones(batch_size,1))))
        d_loss.backward()
        d_optimizer.step()

        if i % 50 == 0:
            print(
                f"step:{len(dataloader) * epoch + i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        if i % 400 == 0:
            image = pred_images[:16].data
            torchvision.utils.save_image(image, f"image_{len(dataloader) * epoch + i}.png", nrow=4)

print('')
