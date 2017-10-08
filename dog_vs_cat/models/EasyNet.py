#coding:utf8
from torch import nn
from .BasicModule import BasicModule
import torch.nn.functional as F

class EasyNet(BasicModule):
    """
    模型类
        简单的卷积神经网络
    """
    def __init__(self, num_classes=2):

        super(EasyNet, self).__init__()
        self.model_name = 'easynet'
        self.features = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size = 3),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size = 3),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size = 3),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
        nn.Linear(25 * 25 * 64, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        x = x.view(x.size(0), 25 * 25 * 64)
        x = self.classifier(x)

        return x
