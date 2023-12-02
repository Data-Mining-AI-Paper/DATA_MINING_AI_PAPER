import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from tool import *
from tf_idf import MyTfidfVectorizer

import numpy as np

def softmax(z):
    z_min = np.min(z)
    z_max = np.max(z)
    normalized_z = (z - z_min) / (z_max - z_min)
    exp_z = np.exp(normalized_z)
    return exp_z / np.sum(exp_z)

print("loading preprocessed data...", end='')
with open("preprocessed_ACL_PAPERS.pickle","rb") as f:
    data = pickle.load(f)
print("Done")

data_by_year =  defaultdict(list)
for paper in data:
    data_by_year[paper.year].append(paper)
data_by_year = dict(sorted(data_by_year.items()))
years = data_by_year.keys()

important_words_by_year = {}
total_keywords = set()
for year, one_year_data in tqdm(data_by_year.items()):
    abstracts = [paper.abstract for paper in one_year_data]

    vectorizer = MyTfidfVectorizer()
    X = vectorizer.fit_transform(abstracts)

    important_words = vectorizer.get_important_words(X, k=20, threshold=0.17)
    # print("important words of paper:", data[0].title)
    # print(*important_words[0].items(), sep='\n')

    important_words_by_year[year] = merge_dict_by_sum(important_words)
    total_keywords |= set(important_words_by_year[year].keys())

num_keywords = [len(important_words_by_year[year].keys()) for year in years]
# Suppose number of keyword of year as number of variety of reasearch topic
num_research_topic = num_keywords
weight_research_topic = np.array(num_research_topic) / np.sum(num_research_topic) 
for i, year in enumerate(data_by_year.keys()):
    for word, tfidf in important_words_by_year[year].items():
        important_words_by_year[year][word] = tfidf / len(data_by_year[year]) * weight_research_topic[i]


from random import sample
keyword_to_visual = []
# keyword_to_visual += sample(sorted(total_keywords),5)
keyword_to_visual += ['translation', 'word', 'knowledge', 'prompt']
print(keyword_to_visual)


## year graph
# tmp = defaultdict(int)
# y, k = [], []
# for key, val in important_words_by_year.items():
#     tmp[key] += len(val.items())
# for a, b in sorted(list(tmp.items())):
#     y.append(a)
#     k.append(b)

# plt.figure(figsize=(10, 6))
# plt.plot(y, k, marker='o', linestyle='-')

# plt.title('trend of N(keys)')
# plt.xlabel('Year')
# plt.ylabel('number of keys')
# plt.grid(True)
# plt.xticks(y, rotation=45)
# plt.tight_layout()

# plt.show()





# print('Trend of keyword: ', end='')
# input_keyword = list(stdin.readline().split())

# while len(input_keyword) != 1:
#     print('input one keyword: ', end='')
#     input_keyword = list(stdin.readline().split())
# input_keyword = input_keyword[0]
# print(important_words_by_year[2016])

keyword_trends = []
for idx, keyword in enumerate(keyword_to_visual):
    keyword_trend = []
    for year in years:
        if keyword in important_words_by_year[year]:
            keyword_trend.append(important_words_by_year[year][keyword])
        else:
            keyword_trend.append(0)
    keyword_trends.append(keyword_trend)
    

plt.figure(figsize=(10, 6))
for i in range(len(keyword_to_visual)):
    plt.plot(years, keyword_trends[i], marker='o', linestyle='-')

plt.title(f"Keyword Trend: {' '.join(keyword_to_visual)}")
plt.xlabel('Year')
plt.ylabel('Value')
plt.grid(True)
# plt.xticks(years, rotation=45)
plt.tight_layout()
plt.legend(keyword_to_visual, loc='best') 

plt.show()
