import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

import torch.optim as optim
import jieba
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from collections import Counter
import os
import re
from tqdm import tqdm
import sys

os.chdir(sys.path[0])

def preprocess_text(text):
    # 删除所有的隐藏符号
    cleaned_text = ''.join(char for char in text if char.isprintable())
    # 删除所有的非中文字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', cleaned_text)
    # 删除所有停用词
    with open('D://yjm//240404//nlp//cn//cn_stopwords.txt', "r", encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
        cleaned_text = ''.join(char for char in cleaned_text if char not in stopwords)
    # 删除所有的标点符号
    punctuation_pattern = re.compile(r'[\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    cleaned_text = punctuation_pattern.sub('', cleaned_text)
    return cleaned_text

def load_data(directory, limit=4):
    """Load data from the specified directory, limited to a certain number of files."""
    corpus = []
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='ansi') as file:
                corpus.append(file.read())
        if i + 1 == limit:
            break
    return corpus

directory = r'C://Users//Administrator//Desktop//pythonProject1//note'
corpus = load_data(directory)

words = [word for text in corpus for word in jieba.lcut(text)]
print(len(words))

counter = Counter(words)
counter['<unk>'] = 0
tokenizer = get_tokenizer('basic_english')
vocab = Vocab(counter)
vocab_size = len(vocab)

words_str = ' '.join(words)
tokens = tokenizer(words_str)
sequences = [vocab[token] for token in tokens]
sequences = [word if word < vocab_size else vocab['<unk>'] for word in sequences]

import torch
from torch import nn
INPUT_SIZE = 50  # 定义输入的特征数
HIDDEN_SIZE = 32    # 定义一个LSTM单元有多少个神经元
BATCH_SIZE = 32   # batch
TIME_STEP = 28   # 步长，一般用不上，写出来就是给自己看的
DROP_RATE = 0.2    #  drop out概率
LAYERS = 2         # 有多少隐层，一个隐层一般放一个LSTM单元
MODEL = 'LSTM'     # 模型名字
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            dropout=DROP_RATE,
            batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
            # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, 6)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        r_out, (h_s, h_c) = self.rnn(x)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output

embedding_dim = 256
hidden_units = 50
model = Seq2Seq(vocab_size, embedding_dim, hidden_units)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


for epoch in tqdm(range(20)):
    optimizer.zero_grad()
    output = model(torch.tensor(sequences[:1000]))
    loss = criterion(output, torch.tensor(sequences[:1000]))
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model.pth')

model = Seq2Seq(vocab_size, embedding_dim, hidden_units)
model.load_state_dict(torch.load('model.pth'))
model.eval()

start_text = "突然间后院马蹄声响，那八名汉子一齐站起，抢出大门。只见"
start_words = list(jieba.cut(start_text))

word2idx = {word: idx for idx, word in enumerate(counter)}
idx2word = {idx: word for idx, word in enumerate(counter)}

start_sequence = [word2idx[word] for word in start_words if word in word2idx]
input = torch.tensor(start_sequence).long().unsqueeze(0)

max_length = 50
generated_sequence = []

for _ in range(max_length):
    output = model(input)
    output_mean = output.mean(dim=1)
    next_word_idx = output_mean.argmax().item()
    generated_sequence.append(next_word_idx)
    input = torch.tensor([next_word_idx]).unsqueeze(0)

generated_words = [idx2word[idx] for idx in generated_sequence]
generated_text = ''.join(generated_words)

print(generated_text)
