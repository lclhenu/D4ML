import torch.nn as nn


class residual(nn.Module):
    def __init__(self):
        super(residual, self).__init__()
        self.cov1_pair1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5))
        self.cov1_1_pair1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.upsample1_pair1 = nn.Upsample(size=(60, 60))
        self.bn1_pair1 = nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True)

        self.cov1_pair2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5))
        self.cov1_1_pair2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.upsample1_pair2 = nn.Upsample(size=(60, 60))
        self.bn1_pair2 = nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True)

        self.cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))
        self.cov2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.upsample2 = nn.Upsample(size=(26, 26))
        self.bn2 = nn.BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True)

        self.cov3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5))
        self.cov3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.upsample3 = nn.Upsample(size=(9, 9))
        self.bn3 = nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True)

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        x1 = self.cov1_pair1(x)  # x1.shape: [batchsize, 32, 60, 60]
        x2 = self.maxpool(x1)  # x2.shape: [batchsize, 32, 30, 30]
        x3 = self.cov1_1_pair1(x2)  # x3.shape: [batchsize, 32, 28, 28]
        x4 = self.upsample1_pair1(x3)  # x4.shape: [batchsize, 32, 60, 60]
        x5 = self.sigmoid(x4)
        x6 = (x5 * x1 + x1)  # x6.shape: [batchsize, 32, 60, 60]
        x7 = self.bn1_pair1(x6)
        x8 = self.relu(x7)
        x9 = self.maxpool(x8)  # x9.shape: [batchsize, 32, 30, 30]

        # 第二层
        x10 = self.cov2(x9)  # x10.shape: [batchsize, 64, 26, 26]
        x11 = self.maxpool(x10)  # x11.shape: [batchsize, 64, 13, 13]
        x12 = self.cov2_2(x11)  # x12.shape: [batchsize, 64, 11, 11]
        x13 = self.upsample2(x12)  # x13.shape: [batchsize, 64, 26, 26]
        x14 = self.sigmoid(x13)
        x15 = (x14 * x10 + x10)  # x15.shape: [batchsize, 64, 26, 26]
        x16 = self.bn2(x15)
        x17 = self.relu(x16)
        x18 = self.maxpool(x17)  # x18.shape: [batchsize, 64, 13, 13]

        # 第三层
        x19 = self.cov3(x18)  # x19.shape: [batchsize, 128, 9, 9]
        x20 = self.maxpool(x19)  # x20.shape: [batchsize, 128, 4, 4]
        x21 = self.cov3_3(x20)  # x21.shape: [batchsize, 128, 2, 2]
        x22 = self.upsample3(x21)  # x21.shape: [batchsize, 128, 9, 9]
        x23 = self.sigmoid(x22)
        x24 = (x23 * x19 + x19)
        x25 = self.bn3(x24)
        x26 = self.relu(x25)  # x21.shape: [batchsize, 128, 9, 9]
        return x26

