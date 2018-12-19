import pandas as pd
import Net
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.size())
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=12):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 1)
        self.layer3 = self.make_layer(block, 64, layers[1], 1)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.shape)
        out = self.avg_pool(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out

batch_size = 100
num_epoches = 100
learning_rate = 0.001

net = ResNet(ResidualBlock, [2, 2, 2, 2])
print(net)
optimizier = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.99))
lossFunction = nn.CrossEntropyLoss()

data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")

y_train = data_train["label"]
y_test = data_test["label"]
data_train.drop("label", axis=1, inplace=True)
data_test.drop("label", axis=1, inplace=True)
X_train = data_train.values
X_test = data_test.values
print(X_train.shape)
print(X_test.shape)

class MyDataSet(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)


train_dataset = MyDataSet(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    # train
    for epoch in range(num_epoches):
        print("epoch: {:d}".format(epoch + 1))
        for data in train_loader:
            img, label = data
            img = img.view(img.size(0), 1, 7, 7)
            # print(img.shape)
            img = img.float()
            img = Variable(torch.Tensor(img))
            label = Variable(torch.LongTensor(label))
            optimizier.zero_grad()
            outputs = net(img)
            # print(outputs)
            _, output = torch.max(outputs, 1)
            # print("label: {:s}, output: {:s}".format(label.data, output))
            loss = lossFunction(outputs, label)
            # print(loss)
            loss.backward()
            optimizier.step()
        if (epoch + 1) % 10 == 0:
            learning_rate /= 3
            optimizier = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # predict
    net.eval()
    X_test = Variable(torch.Tensor(X_test.reshape(-1, 1, 7, 7)), volatile=True)
    print(X_test.data.shape)
    out = net(X_test)
    _, pred = torch.max(out, 1)
    y_test = Variable(torch.LongTensor(y_test))
    # print(pred)
    # print(y_test)
    num_correct = (pred.data == y_test.data).sum()
    print(num_correct)
    print("Accuracy : " + (float(num_correct) / y_test.data.shape[0]).__str__())
    with open("submission.csv", "w") as f:
        f.write("ImageId,Label\n")
        count = 1
        count2 = 1
        for i in pred:
            f.write(count.__str__() + "," + i.data[0].__str__() + ", " + y_test.data[count-1].__str__())
            if i.data[0] == y_test.data[count-1]:
                # f.write(", " + count2.__str__())
                count2 = count2 + 1
            f.write("\n")
            count += 1
