import os
import jieba
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import pearsonr
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
# 计算皮尔逊相关指数，并进行显著性检验
def get_pex(x,y):
    aaa = scipy.stats.pearsonr(x, y)
    print('皮尔逊系数是：{}，显著性检测值是：{}'.format(aaa[0],aaa[1]))

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

#plt.rcParams['font.sans-serif'] = ['SimHei']
folder_path = r'D://yjm//240404//nlp'
datas = []

j=0
for file_name in os.listdir(folder_path):
    j+=1
    if file_name.endswith(".txt"):
        print(file_name)
        file_path = os.path.join(folder_path,file_name)
        with open(file_path,"r",encoding='ansi') as file:
            text = file.read()
            text=preprocess_text(text)
            words = jieba.lcut(text)
        # 去掉长度为1的词，包括标点
        newls = []
        for i in words:
            if len(i) >=1:
                newls.append(i)
        # 统计词频
        ds = pd.Series(newls).value_counts()
        #plt.subplots((int)(j%4+1),(int)(j/3+1))

        plt.figure(figsize=(10, 6))
        y1 = []
        for i in range(len(ds)):
          feature = [float(ds.iloc[i])]
          feature = math.log(feature[0])
          y1.append(feature)
#print(y)
        y=np.array(y1)
        y2=y
        x1=np.arange(1,len(y)+1)
        x1 = x1.astype(float)
        for i in range(len(x1)):
          x1[i]=float(math.log(x1[i]))

        x=x1.reshape(-1,1)
        y=y.reshape(-1,1)
        model = LinearRegression()
        model.fit(x,y)
        plt.ylabel('log(Frequency)',fontsize=30)
        plt.xlabel('log(Rank)',fontsize=30)
        plt.title(file_name,fontsize=30)
        plt.scatter(x,y)
        plt.plot(x,model.predict(x),color='red')

plt.show()
