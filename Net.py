from torch import nn


class Net(nn.Module):
    def __init__(self, classesNum):
        super(Net, self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1))
        layer1.add_module('relu1', nn.ReLU(inplace=True))
        layer1.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1))
        layer2.add_module('relu2', nn.ReLU(inplace=True))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1))
        layer3.add_module('relu3', nn.ReLU(inplace=True))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('conv4', nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1))
        self.layer4 = layer4

        self.feature = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )

        self.classfier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(16 * 3 * 3, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, classesNum),
            # nn.Softmax()
        )

    def forward(self, x):
        # 100 * 1 * 7 * 7
        x = self.feature(x)
        # 100 * 16 * 3 * 3
        x = x.view(x.size(0), 16 * 3 * 3)
        # 100 * 144
        x = self.classfier(x)
        # 100 * 12
        return x


