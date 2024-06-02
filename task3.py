#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import gensim
import numpy as np
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
#读取文件
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
folder_path = r'D://yjm//240507//assignment2'
corpus = []#存储分词后的文本
for file_name in os.listdir(folder_path):
  if file_name.endswith(".txt"):
    file_path = os.path.join(folder_path, file_name)
  with open(file_path, 'r', encoding='ansi') as f:
    file_read = f.readlines()
    all_text = " "
    for line in file_read:
        line = line.strip('\n')
        #line = re.sub("[A-Za-z0-9\：\·\—\，\。\“\”\\n \《\》\！\？\、\...]", "", line)
        all_text += line
        text = preprocess_text(line)
        con = jieba.cut(line, cut_all=False)
        corpus.append(list(con))

def content_deal(content):
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        content = content.replace(a, '')
    return content
def get_paragraph_vector(paragraph, model):
    # 分词并去除停用词
    with open('D://yjm//240404//nlp//cn//cn_stopwords.txt', "r", encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    words = [word for word in jieba.cut(paragraph) if word not in stopwords]
    # 计算词向量
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    # 取平均值作为段落向量
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
def calculate_similarity(paragraph1, paragraph2, model):
    vector1 = get_paragraph_vector(paragraph1, model)
    vector2 = get_paragraph_vector(paragraph2, model)
    if np.all(vector1 == 0) or np.all(vector2 == 0):
        return 0  # 若有一个段落无有效向量，则相似度为0
    else:
        return cosine_similarity([vector1], [vector2])[0][0]
#建立Word2Vec模型，并构建词汇表
model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
model.build_vocab(corpus)

#训练Word2Vec模型
model.train(corpus, total_examples=model.corpus_count, epochs=10)

#导出“刀光”的词向量
mingjiao_vector = model.wv['刀光']

#导出与“刀光”最相关的前20个词语
similar_words = model.wv.most_similar('刀光', topn=10)
print("刀光的词向量:", mingjiao_vector)
print("\n与刀光最相关的前20个词语")
for word, similarity in similar_words:
    print(f"{word}:{similarity}")
qiaofeng_vector = model.wv['虚竹']
n_clusters=800
#导出与“虚竹”最相关的前20个词语
similar_words = model.wv.most_similar('虚竹', topn=10)
print("虚竹的词向量:", qiaofeng_vector)
print("\n与虚竹最相关的前10个词语")
for word, similarity in similar_words:
    print(f"{word}:{similarity}")
# 获取词汇表中的所有词语及其向量
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])

# 使用KMeans算法进行聚类
num_clusters = 10  # 设定要分的簇数
kmeans = KMeans(n_clusters=num_clusters, n_init= 'auto',random_state=0).fit(word_vectors)

# 获取每个词语对应的簇标签
labels = kmeans.labels_

# 获取 Cluster 1 的词向量和索引
cluster_index = 1
cluster_indices = [index for index, label in enumerate(labels) if label == cluster_index]
cluster_words = [words[index] for index in cluster_indices]
cluster_vectors = word_vectors[cluster_indices]

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
cluster_vectors_2d = tsne.fit_transform(cluster_vectors)

# 绘制散点图
plt.figure(figsize=(8, 8))
plt.scatter(cluster_vectors_2d[:, 0], cluster_vectors_2d[:, 1], label=f'Cluster {cluster_index}')


plt.legend()
plt.title(f'Word Clustering Visualization for Cluster {cluster_index}')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
clusters = {}
for i in range(num_clusters):
    clusters[i] = []

for word, label in zip(words, labels):
    clusters[label].append(word)

for cluster_id, cluster_words in clusters.items():
    print(f"Cluster {cluster_id}: {', '.join(cluster_words)}")
paragraph1 = "林震南点头道：“老头儿怕事，这里杀伤了人命，尸体又埋在他菜园子里，他怕受到牵连，就此一走了之。”走到菜园里，指着倚在墙边的一把锄头，说道：“陈七，把死尸掘出来瞧瞧。”陈七早认定是恶鬼作祟，只锄得两下，手足俱软，直欲瘫痪在地。季镖头道：“有个屁用？亏你是吃镖行饭的！”一手接过锄头，将灯笼交在他手里，举锄扒开泥土，锄不多久，便露出死尸身上的衣服，又扒了几下，将锄头伸到尸身下，用力一挑，挑起死尸。陈七转过了头，不敢观看，却听得四人齐声惊呼，陈七一惊之下，失手抛下灯笼，蜡烛熄灭，菜园中登时一片漆黑。林平之颤声道：“咱们明明埋的是那四川人，怎地……怎地……”林震南道：“快点灯笼！”他一直镇定，此刻语音中也有了惊惶之意。崔镖头晃火折点着灯笼，林震南弯腰察看死尸，过了半晌，道：“身上也没伤痕，一模一样的死法。”陈七鼓起勇气，向死尸瞧了一眼，尖声大叫：“史镖头，史镖头！”地下掘出来的竟是史镖头的尸身，那四川汉子的尸首却已不知去向。林震南道：“这姓萨的老头定有古怪。”抢着灯笼，奔进屋中察看，从灶下的酒坛、铁镬，直到厅房中的桌椅都细细查了一遍，不见有异。崔季二镖头和林平之也分别查看。突然听得林平之叫道：“咦！爹爹，你来看。”"
paragraph2 = "林震南循声过去，见儿子站在那少女房中，手中拿着一块绿色帕子。林平之道：“爹，一个贫家女子，怎会有这种东西？”林震南接过手来，一股淡淡幽香立时传入鼻中，那帕子甚是软滑，沉甸甸的，显是上等丝缎，再一细看，见帕子边缘以绿丝线围了三道边，一角上绣着一枝小小的红色珊瑚枝，绣工甚是精致。林震南问：“这帕子哪里找出来的？”林平之道：“掉在床底下的角落里，多半是他们匆匆离去，收拾东西时没瞧见。”林震南提着灯笼俯身又到床底照着，不见别物，沉吟道：“你说那卖酒的姑娘相貌甚丑，衣衫质料想来不会华贵，但是不是穿得十分整洁？”林平之道：“当时我没留心，但不见得污秽，倘若很脏，她来斟酒之时我定会觉得。"
# 计算段落相似度
similarity_score = calculate_similarity(paragraph1, paragraph2, model)
print(f"段落1与段落2的语义相似度：{similarity_score}")

import os
import re
import gensim
import numpy as np
import jieba
#读取文件
file_path = 'D://yjm//240507//assignment2//天龙八部.txt'
corpus = []#存储分词后的文本

with open(file_path, 'r', encoding='ansi') as f:
    file_read = f.readlines()
    all_text = " "
    for line in file_read:
        line = line.strip('\n')
        line = re.sub("[A-Za-z0-9\：\·\—\，\。\“\”\\n \《\》\！\？\、\...]", "", line)
        all_text += line
        con = jieba.cut(line, cut_all=False)
        corpus.append(list(con))

def content_deal(content):
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        content = content.replace(a, '')
    return content

#建立Word2Vec模型，并构建词汇表
model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
model.build_vocab(corpus)

#训练Word2Vec模型
model.train(corpus, total_examples=model.corpus_count, epochs=10)

#导出“虚竹”的词向量
qiaofeng_vector = model.wv['虚竹']

#导出与“虚竹”最相关的前10个词语
similar_words = model.wv.most_similar('虚竹', topn=10)
print("虚竹的词向量:", qiaofeng_vector)
print("\n与虚竹最相关的前10个词语")
for word, similarity in similar_words:
    print(f"{word}:{similarity}")


# In[ ]:
