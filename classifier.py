import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sympy import jn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
import csv
# 设置随机数种子保证论文可复现
seed = 42
num1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
data_train0 = pd.read_csv('D://yjm//240507//assignment1//train.csv')
data_test0 = pd.read_csv('D://yjm//240507//assignment1//test.csv')

data_train1=data_train0.copy(deep=True)
data_test1=data_test0.copy(deep = False)
# print(all_features)
numeric_features = data_train1.dtypes[data_train1.dtypes != 'object'].index
data_train1[numeric_features] = data_train1[numeric_features].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
data_train1[numeric_features] = data_train1[numeric_features].fillna(0)
numeric_features = data_test1.dtypes[data_test1.dtypes != 'object'].index
data_test1[numeric_features] = data_test1[numeric_features].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
data_test1[numeric_features] = data_test1[numeric_features].fillna(0)
n_train = data_train1.shape[0]
data_train2=np.array(data_train1)
data_test2=np.array(data_test1)
class Args:
    def __init__(self) -> None:
        self.batch_size = 10
        self.lr = 0.001
        self.epochs = 30
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_train =data_train2[:,0:len(data_train2[1])-1]
        #print(len(data_train1[1])-1)
        self.data_val = data_test2[:,0:len(data_test2[1])-1]
        self.train_labels = data_train0.iloc[:,-1]
        self.test_labels = data_test0.iloc[:, -1]
        #print(self.test_labels)
args = Args()


class SelfAttrntion(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttrntion, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weight = torch.matmul(q, k.transpose(1, 2))
        attn_weight = nn.functional.softmax(attn_weight, dim=-1)
        attned_values = torch.matmul(attn_weight, v)
        return attned_values


# 定义一个简单的全连接
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.attention = SelfAttrntion(in_dim)
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True), nn.Dropout(0.3))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True), nn.Dropout(0.3))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.attention(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer4(x)
        return x


class Dataset_num(Dataset):
    def __init__(self, flag='train') -> None:
        self.flag = flag
        assert self.flag in ['train', 'val'], 'not implement!'

        if self.flag == 'train':
            self.data = args.data_train
        else:
            self.data = args.data_val

    def __getitem__(self, index: int):
        val = self.data[index]
        if self.flag == 'train':
            label = args.train_labels[index]
        else:
            label = args.test_labels[index]
        return torch.tensor(label, dtype=torch.long), torch.tensor([val], dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.data))


def train():
    train_dataset = Dataset_num(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = Dataset_num(flag='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    model = Net(50, 128, 64, 17).to(args.device)  # 网路参数设置，输入为1，输出为2，即判断一个数是否大于8
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # , eps=1e-8)

    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        # =========================train=======================
        for idx, (label, inputs) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(args.device)
            #print(inputs.size())
            label = label.to(args.device)
            outputs = model(inputs)

            outputs = torch.squeeze(outputs, dim=1)
            # print(outputs.size())
            # print(outputs.size())
            optimizer.zero_grad()
            #label = torch.squeeze(label, dim=1)
            # print(label.size())
            loss = criterion(outputs, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) #用来梯度裁剪
            optimizer.step()
            train_epoch_loss.append(loss.item())
            # print(label.eq(outputs.max(axis=1)[1]))
            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            # print(outputs.max(axis=1)[1])
            # print('-------------------------------')
            # print(label)
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}%, loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))
        # print(num1)
        # C1=confusion_matrix
        # plt.figure(figsize=(12,8))
        # plt.tight_layout()
        # sns.heatmap
        # plt.show()
        # =========================val=========================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0

            for idx, (label, inputs) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.to(args.device)  # .to(torch.float)
                #print(inputs.size())
                label = label.to(args.device)
                outputs = model(inputs)
                outputs = torch.squeeze(outputs, dim=1)
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())

                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]
            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)

            print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(epoch, 100 * acc / nums,
                                                                      np.average(val_epoch_loss)))

    # =========================plot==========================
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(train_epochs_loss[:])
    plt.title("train_loss")
    plt.subplot(132)
    plt.plot(train_epochs_loss, '-o', label="train_loss")
    plt.plot(valid_epochs_loss, '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.show()
    # =========================save model=====================
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    train()
# 这个会预测错误，所以数据量对于深度学习很重要
