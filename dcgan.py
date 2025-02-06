"""
输入图片必须为正方形，若想要保留长方形训练，可选择裁切or填补方案。
若想要保留整个图片，可将 加载数据集 的 CustomCrop()注释掉，反之则会裁剪成正方形
裁剪原则在CustomCrop类那儿，可以修改尺寸啥的
"""
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

# 自定义填充函数：将图片填充为正方形
def pad_to_square(img, fill=0):
    """
    将图片填充为正方形：
    1. 如果图片是长方形，在短边两侧填充像素。
    2. 填充颜色默认黑色（fill=0），可改为其他颜色（如 fill=255 为白色）。
    """
    w, h = img.size
    if w == h:
        return img
    elif w > h:
        # 高度不足，填充上下
        padding = (0, (w - h) // 2, 0, (w - h) - (w - h) // 2)  # (左, 上, 右, 下)
    else:
        # 宽度不足，填充左右
        padding = ((h - w) // 2, 0, (h - w) - (h - w) // 2, 0)
    return transforms.functional.pad(img, padding, fill=fill)

# 设置参数
"""
dataroot - 数据集存放路径. 我们将在下一节中深入讨论
workers - 多进程加载数据所用的进程数
batch_size - 训练时batch的大小.  DCGAN 论文中使用的是 128
image_size -训练图片的尺寸. 这里默认是 64x64.如果需要另一种尺寸,则必须更改 D 和G 的结构
nc - 输入图片的通道数. 这里是3
nz - 隐向量的维度（即来自标准正态分布的隐向量的维度）（也即高斯噪声的维度）
ngf - 生成器的特征图数量（即进行最后一次卷积转置层时，out_channels为3时的in_channels）
ndf - 判别器的特征图数量（即进行第一次卷积时，in_channels为3时的out通道数）
num_epochs - 训练模型的迭代次数。长时间的训练可能会带来更好的结果，但也需要更长的时间
lr - 训练时的学习率. 在DCGAN 论文中, 这个数值是 0.0002
beta1 - Adam 优化器的beta1参数. 在论文中,此数值是0.5
ngpu - 可用GPU的数量. 如果为 0, 代码将使用CPU训练. 如果大于0，将使用此数值的GPU进行训练
"""
class Options:
    def __init__(self):
        self.dataset = "folder"
        self.dataroot = "train_images"  # 数据集路径
        self.batchSize = 370
        self.imageSize = 256  # 目标尺寸（正方形）
        self.nz = 100
        self.ngf = 64
        self.ndf = 64
        self.niter = 450  # 训练轮数
        self.lr = 0.00035
        self.beta1 = 0.5
        self.cuda = True
        self.ngpu = 1
        self.netG = ""
        self.netD = ""
        self.outf = "results"
        self.manualSeed = None
        self.classes = None
        self.workers = 0
        self.nc = 3

opt = Options()

# 设置随机种子
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 创建输出文件夹
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


class CustomCrop:
    def __call__(self, img):
        width, height = img.size
        # 计算裁剪高度（40% 的高度）
        crop_height = int(height * 0.35)
        # 计算正方形边长（不能超过原图的宽度）
        crop_size = min(crop_height, width)
        # 计算左侧和右侧，使裁剪区域居中
        left = (width - crop_size) // 2
        right = left + crop_size
        # 上方从 0 开始，下方是 crop_size
        top = 0
        bottom = crop_size
        return img.crop((left, top, right, bottom))


# 加载数据集
transform = transforms.Compose([
    CustomCrop(),  # 自定义裁剪*
    transforms.Lambda(lambda img: pad_to_square(img, fill=0)),  # 填充为正方形
    transforms.Resize((opt.imageSize, opt.imageSize)),  # 缩放到目标尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = dset.ImageFolder(
    root=opt.dataroot,
    transform=transform
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers)
)

# 定义生成器和判别器（保持与之前一致）
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x.view(-1)

# 初始化模型、优化器等（保持与之前一致）
netG = Generator().to(device)
netD = Discriminator().to(device)



criterion = nn.BCEWithLogitsLoss()  # 替代 nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # /降低判别器的学习率


# （打印一下调试信息）
# 检查数据集大小
print(f"数据集大小: {len(dataset)}")
# 检查数据加载器批次数量
print(f"数据加载器批次数量: {len(dataloader)}")
# 检查批量大小
print(f"批量大小: {opt.batchSize}")


# 循环
# 断点续传，继续迭代之前的权重
# netG.load_state_dict(torch.load("results.1/netG_epoch_79.pth"))
# netD.load_state_dict(torch.load("results.1/netD_epoch_79.pth"))
# 训练循环（for循环接断点续传）
for epoch in range(opt.niter):
    for i, (data, _) in enumerate(dataloader):
        # 训练判别器
        netD.zero_grad()
        real_data = data.to(device)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), 1.0, device=device)
        output = netD(real_data)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_data = netG(noise)
        label.fill_(0.0)
        output = netD(fake_data.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        label.fill_(1.0)
        output = netD(fake_data)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        # 打印训练状态
        #if i % 50 == 0:
        print(f"[{epoch}/{opt.niter}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

        # 保存生成图片
        #if i % 100 == 0:
        vutils.save_image(real_data, f"{opt.outf}/real_samples.png", normalize=True)
        fake = netG(torch.randn(batch_size, nz, 1, 1, device=device))
        vutils.save_image(fake.detach(), f"{opt.outf}/fake_samples_epoch_{epoch}.png", normalize=True)

    # 保存模型
    torch.save(netG.state_dict(), f"{opt.outf}/netG_epoch_{epoch}.pth")
    torch.save(netD.state_dict(), f"{opt.outf}/netD_epoch_{epoch}.pth")
