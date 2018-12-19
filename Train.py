import pandas as pd
import Net
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data

batch_size = 100
num_epoches = 100
learning_rate = 0.001

net = Net.Net(classesNum=12)
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
