import scipy.stats

import jieba
import os
import re
import numpy as np
from collections import Counter
import copy
from collections import Counter
import math
from sklearn.linear_model import LinearRegression
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
def calculate_entropy(text):
    frequencies = Counter(text)
    total_length = len(text)
    entropy = 0
    for char, freq in frequencies.items():
        probability = freq / total_length
        entropy -= probability * math.log2(probability)
    return entropy

def get_bigram_tf(word):
        # 得到二元词的词频表
    bigram_tf = {}
    for i in range(len(word) - 1):
        bigram_tf[(word[i], word[i + 1])] = bigram_tf.get(
        (word[i], word[i + 1]), 0) + 1
    return bigram_tf

def get_trigram_tf( word):

        # 得到三元词的词频表
    trigram_tf = {}
    for i in range(len(word) - 2):
        trigram_tf[(word[i], word[i + 1], word[i + 2])] = trigram_tf.get(
            (word[i], word[i + 1], word[i + 2]), 0) + 1
    return trigram_tf


def calc_entropy_bigram( word, is_ci):
            # 计算二元模型的信息熵
            # 计算二元模型总词频
    word_tf = get_bigram_tf(word)
    last_word_tf = get_unigram_tf(word)
    bigram_len = sum([item[1] for item in word_tf.items()])
    entropy = []
    for bigram in word_tf.items():
        p_xy = bigram[1] / bigram_len  # 联合概率p(xy)
        p_x_y = bigram[1] / last_word_tf[bigram[0][0]]  # 条件概率p(x|y)
        entropy.append(-p_xy * math.log(p_x_y, 2))
    entropy = sum(entropy)
    if is_ci:
        print("基于词的二元模型的中文信息熵为：{}比特/词".format( entropy))
    else:
        print("基于字的二元模型的中文信息熵为：{}比特/词".format( entropy))
    return entropy
def calculate_word_entropy(text):
    words = text.split()
    total_words = len(words)

    entropy_sum = 0
    for word in words:
        entropy_sum += calculate_entropy(word)
    return entropy_sum / total_words

def calculate_character_entropy(text):
    total_characters = len(text)
    return calculate_entropy(text) / total_characters


def calc_entropy_trigram(word, is_ci):
        # 计算三元模型的信息熵
        # 计算三元模型总词频
    word_tf = get_trigram_tf(word)
    last_word_tf = get_bigram_tf(word)
    trigram_len = sum([item[1] for item in word_tf.items()])
    entropy = []
    for trigram in word_tf.items():
        p_xy = trigram[1] / trigram_len  # 联合概率p(xy)
        p_x_y = trigram[1] / last_word_tf[(trigram[0][0], trigram[0][1])]  # 条件概率p(x|y)
        entropy.append(-p_xy * math.log(p_x_y, 2))
    entropy = sum(entropy)
    if is_ci:
        print("基于词的三元模型的中文信息熵为：{}比特/词".format(entropy))
    else:
            print("基于字的三元模型的中文信息熵为：{}比特/字".format( entropy))
    return entropy


def get_unigram_tf(word):
    # 得到单个词的词频表
    unigram_tf = {}
    for w in word:
        unigram_tf[w] = unigram_tf.get(w, 0) + 1
    return unigram_tf
def calc_entropy_unigram(word1, is_ci):
    # 计算一元模型的信息熵
    word_tf = get_unigram_tf(word1)
    word_len = sum([item[1] for item in word_tf.items()])
    entropy = sum(
        [-(word[1] / word_len) * math.log(word[1] / word_len, 2) for word in
        word_tf.items()])
    if is_ci:
        print("基于词的一元模型的中文信息熵为：{}比特/词".format( entropy))
    else:
        print("基于字的一元模型的中文信息熵为：{}比特/字".format( entropy))
    return entropy
#plt.rcParams['font.sans-serif'] = ['SimHei']
folder_path = r'D://yjm//240404//nlp'
datas = []
newls = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path,file_name)
        with open(file_path,"r",encoding='ansi') as file:
            text = file.read()
            text=preprocess_text(text)
            words = jieba.lcut(text)
            print(file_name)
            #character_entropy = calc_entropy_unigram(words,1)
            #character_entropy1 = calc_entropy_bigram(words, 1)
            character_entropy1 = calc_entropy_trigram(text, 0)
#print(word_entropy,character_entropy)


