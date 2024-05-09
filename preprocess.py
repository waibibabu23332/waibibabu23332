import os
import jieba
import math
import re
import pyLDAvis.gensim_models
import jieba.posseg as jp
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import pearsonr
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import csv
import random
import pandas
from sklearn.svm import SVC
# 计算皮尔逊相关指数，并进行显著性检验
def Dataset(content):
    traindata, trainlabel = [], []
    testdata, testlabel = [], []
    random.shuffle(content) # 打乱数据集
    j=0
    for i in range(int(len(content) * 0.9)):
     for grade in content[i].values():
      if(j%3==1):
       trainlabel.append(grade)
      if(j%3==2):
       traindata.append(grade)
      j=j+1
      if(j==3):
       j=0
    j=0
    for i in range(int(len(content)*0.9), int(len(content))):  # 剩下10%做测试集
        for grade in content[i].values():
            if (j % 3 == 1):
                testlabel.append(grade)
            if (j % 3 == 2):
                testdata.append(grade)
            j = j + 1
            if (j == 3):
                j = 0
    return traindata, trainlabel,testdata, testlabel
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
is_word =1
folder_path = r'D://yjm//240507//assignment2'
datas = []
newls = []
j=0
dict = {}
word_len = 0

for file_name in os.listdir(folder_path):

    j+=1
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path,file_name)
        with open(file_path,"r",encoding='ansi') as file:
            text = file.read()
            text=preprocess_text(text)
            words = jieba.lcut(text)
        # 去掉长度为1的词，包括标点
        newls1 = []
        for i in words:
            if len(i) >=2:#这里设置的是字
                newls.append(i)
                newls1.append(i)
        dict[file_name]=newls1
        word_len=word_len+len(dict[file_name])

con_list = []
number= 1
for file_name in os.listdir(folder_path):
    count = int(len(dict[file_name])/word_len*1000+0.5) #总共抽取200个段落

    pos = int(len(dict[file_name])//count)#抽取段落
    for i in range(count):
     data_temp = dict[file_name][i*pos:i*pos+100]
     con ={
        'number':number,
        'label':file_name,
         'data':data_temp
     }
     con_list.append(con)
     number = number+1
if(is_word):
    save_path = 'D://yjm//240507//assignment1//word.csv'
else:
    save_path = 'D://yjm//240507//assignment1//words.csv'
with open(save_path, 'a', newline='', encoding='GB18030') as fp:
    csv_header = ['number', 'label', 'data']  # 设置表头，即列名
    #csv_writer = csv.DictWriter(fp, csv_header)
    #if fp.tell() == 0:
    #    csv_writer.writeheader()
    #csv_writer.writerows(con_list)  # 写入数据
dictionary = Dictionary([newls])
corpus = [dictionary.doc2bow(words1) for words1 in [newls]]
num_topics1 =50#设置主题数

lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics1 , random_state=100, iterations=10)
ldaCM = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
[traindata,trainlabel,testdata,testlabel]=Dataset(con_list)
for i in range(len(trainlabel)):
    if(trainlabel[i]=='白马啸西风.txt'):
        trainlabel[i] = 1
    if (trainlabel[i] == '碧血剑.txt'):
        trainlabel[i] = 2
    if (trainlabel[i] == '飞狐外传.txt'):
        trainlabel[i] = 3
    if (trainlabel[i] == '连城诀.txt'):
        trainlabel[i] = 4
    if (trainlabel[i] == '鹿鼎记.txt'):
        trainlabel[i] = 5
    if (trainlabel[i] == '射雕英雄传.txt'):
        trainlabel[i] = 6
    if (trainlabel[i] == '三十三剑客图.txt'):
        trainlabel[i] = 7
    if (trainlabel[i] == '神雕侠侣.txt'):
        trainlabel[i] = 8
    if (trainlabel[i] == '书剑恩仇录.txt'):
        trainlabel[i] = 9
    if (trainlabel[i] == '天龙八部.txt'):
        trainlabel[i] = 10
    if (trainlabel[i] == '侠客行.txt'):
        trainlabel[i] = 11
    if (trainlabel[i] == '笑傲江湖.txt'):
        trainlabel[i] = 12
    if (trainlabel[i] == '倚天屠龙记.txt'):
        trainlabel[i] = 13
    if (trainlabel[i] == '鸳鸯刀.txt'):
        trainlabel[i] = 14
    if (trainlabel[i] == '越女剑.txt'):
        trainlabel[i] = 15
    if (trainlabel[i] == '雪山飞狐.txt'):
        trainlabel[i] = 16
for i in range(len(testlabel)):
    if(testlabel[i]=='白马啸西风.txt'):
        testlabel[i] = 1
    if (testlabel[i] == '碧血剑.txt'):
        testlabel[i] = 2
    if (testlabel[i] == '飞狐外传.txt'):
        testlabel[i] = 3
    if (testlabel[i] == '连城诀.txt'):
        testlabel[i] = 4
    if (testlabel[i] == '鹿鼎记.txt'):
        testlabel[i] = 5
    if (testlabel[i] == '射雕英雄传.txt'):
        testlabel[i] = 6
    if (testlabel[i] == '三十三剑客图.txt'):
        testlabel[i] = 7
    if (testlabel[i] == '神雕侠侣.txt'):
        testlabel[i] = 8
    if (testlabel[i] == '书剑恩仇录.txt'):
        testlabel[i] = 9
    if (testlabel[i] == '天龙八部.txt'):
        testlabel[i] = 10
    if (testlabel[i] == '侠客行.txt'):
        testlabel[i] = 11
    if (testlabel[i] == '笑傲江湖.txt'):
        testlabel[i] = 12
    if (testlabel[i] == '倚天屠龙记.txt'):
        testlabel [i]= 13
    if (testlabel[i] == '鸳鸯刀.txt'):
        testlabel[i] = 14
    if (testlabel[i] == '越女剑.txt'):
        testlabel [i]= 15
    if (testlabel[i] == '雪山飞狐.txt'):
        testlabel [i]= 16

train_features = np.zeros((len(traindata), num_topics1))
lda_corpus_train = [dictionary.doc2bow(tmp_doc) for tmp_doc in traindata]
train_topic_distribution = lda.get_document_topics(lda_corpus_train)
for i in range(len(train_topic_distribution)):
        tmp_topic_distribution = train_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            train_features[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]
assert len(trainlabel) == len(train_features)
train_label = np.array(trainlabel)
train_label=train_label.T
lda_corpus_test = [dictionary.doc2bow(tmp_doc) for tmp_doc in testdata]
test_topic_distribution = lda.get_document_topics(lda_corpus_test)
test_features = np.zeros((len(testdata), num_topics1))
test_label = np.array(testlabel)
test_label=test_label.T
train_features=np.insert(train_features,num_topics1,trainlabel,axis=1)
for i in range(len(test_topic_distribution)):
    tmp_topic_distribution = test_topic_distribution[i]
    for j in range(len(tmp_topic_distribution)):
        test_features[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]
assert len(testlabel) == len(test_features)
test_features=np.insert(test_features,num_topics1,testlabel,axis=1)
save_path = 'D://yjm//240507//assignment1//train.csv'
with open(save_path, 'a', newline='', encoding='GB18030') as fp:
    csv_writer = csv.writer(fp)
    for i in range(len(train_features)):
     csv_writer.writerow(train_features[i])
save_path = 'D://yjm//240507//assignment1//test.csv'
with open(save_path, 'a', newline='', encoding='GB18030') as fp:
    csv_writer = csv.writer(fp)
    for i in range(len(test_features)):
        csv_writer.writerow(test_features[i])
