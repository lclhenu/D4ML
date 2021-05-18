from residual_block import *
import torch
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x



class train_Net(nn.Module):
    def __init__(self):
        super(train_Net, self).__init__()
        self.part = 3  # We cut the pool5 to 6 parts
        # self.part_avgpool = nn.AdaptiveMaxPool2d((self.part, 1))
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(3456, 400, droprate=0.5))
        self.model = residual().cuda()
        # self.classifier = ClassBlock(10368, 533, 0.5)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.avgpool = nn.AvgPool1d(2, stride=2)
        self.last1 = nn.Sequential(
            nn.Linear(41472, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1)
        )
        self.last1.apply(weights_init_classifier)

    # [batchsize, channel, width, height]
    def forward(self, x1, x2):  # x.shape: [batchsize, 6, 64, 64]
        # 第一层
        feature1 = self.model(x1)
        feature2 = self.model(x2)

        # f1 = self.classifier(feature1.view(feature1.size(0), -1))
        # f2 = self.classifier(feature2.view(feature2.size(0), -1))
        f = torch.cat((feature1, feature2), dim=0)
        # f = self.part_avgpool(f)
        f = f.view(f.size(0), -1, 3, 1)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(f[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.part):
            y.append(predict[i])

        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        x = self.last1(x)

        return x, feature1, feature2, f, y





