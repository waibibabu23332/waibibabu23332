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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
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
def finetune_model(model, train_dataset, output_dir, tokenizer, model_type="gpt2"):
    """微调模型"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()


torch.cuda.is_available = lambda: False

# 加载预训练模型和分词器
gpt2_model_path = "uer/gpt2-distil-chinese-cluecorpussmall"

gpt2_model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall").to(torch.device('cuda:0'))
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)

# 预处理中文语料库
corpus_file = r'C://Users//Administrator//Desktop//pythonProject1//note//鹿鼎记.txt'
chinese_corpus = preprocess_chinese_corpus(corpus_file)

# 加载数据集
train_dataset_gpt2 = load_dataset_from_corpus(chinese_corpus, gpt2_tokenizer, model_type="gpt2")

finetune_model(gpt2_model, train_dataset_gpt2, "./gpt2-finetuned", gpt2_tokenizer, model_type="gpt2")


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
