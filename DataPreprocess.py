import pandas as pd
import numpy as np

output = pd.read_csv("data/output.csv")
source = pd.read_table("data/source.txt", header=None)

y = output["Protocol"]
dict = {}
t = 0
for i in y:
    if(dict.get(i) == None):
        dict[i] = t
        t = t + 1
print(dict)
label = np.zeros(y.__len__(), int)
for i in range(y.__len__()):
    label[i] = dict[y.ix[i]]
# print(label)

# print(source)
contains = []
for i in range(source.__len__()):
    if (i + 1) % 3 == 0:
        s = np.array(source.ix[i])
        # print(s[0])
        contains.append(s[0])
print(contains.__len__())

Res = []
for contain in contains:
    # print(contain.__len__())
    temp = contain.split('|')
    # print(temp)
    res = []
    for i in range(temp.__len__() - 1):
        if(i >= 2 and i <= 50):
            temp2 = temp[i]
            number = int(temp2, 16)
            res.append(number)
    if(res.__len__() < 49):
        # print(res.__len__())
        while res.__len__() != 49:
            res.append(0)
    if(res.__len__() != 49):
        print(res.__len__())
    # print(res)
    Res.append(res)

data = pd.DataFrame()
label = pd.Series(label)
data["label"] = label
# print(data)
for i in range(49):
    b = [x[i] for x in Res]
    data["pixel" + str(i)] = b
print(data.head())
data.to_csv("data/data.csv", sep=",", header=True, index=False)
data[0:5000].to_csv("data/train.csv", sep=",", header=True, index=False)
data[5000:].to_csv("data/test.csv", sep=",", header=True, index=False)

