import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud
from collections import defaultdict
from sys import stdin

from tool import *
from tf_idf import MyTfidfVectorizer

print("loading preprocessed data...", end='')
with open("preprocessed_ACL_PAPERS.pickle","rb") as f:
    data = pickle.load(f)
print("Done")

data_by_year =  defaultdict(list)
for paper in data:
    data_by_year[paper.year].append(paper)

important_words_by_year = {}
keyword_trend_dict = {}

for year, one_year_data in tqdm(data_by_year.items()):
    abstracts = [paper.abstract for paper in one_year_data]

    vectorizer = MyTfidfVectorizer()
    X = vectorizer.fit_transform(abstracts)

    important_words = vectorizer.get_important_words(X, k=20, threshold=0.2)
    # print("important words of paper:", data[0].title)
    # print(*important_words[0].items(), sep='\n')

    important_words_by_year[year] = merge_dict_by_sum(important_words)
    for key, val in important_words_by_year[year].items():
        important_words_by_year[year][key] = val / len(one_year_data) * len(important_words_by_year[year].keys())


## year graph
tmp = defaultdict(int)
y, k = [], []
for key, val in important_words_by_year.items():
    tmp[key] += len(val.items())
for a, b in sorted(list(tmp.items())):
    y.append(a)
    k.append(b)

plt.figure(figsize=(10, 6))
plt.plot(y, k, marker='o', linestyle='-')

plt.title('trend of N(keys)')
plt.xlabel('Year')
plt.ylabel('number of keys')
plt.grid(True)
plt.xticks(y, rotation=45)
plt.tight_layout()

plt.show()


print('Trend of keyword: ', end='')
input_keyword = list(stdin.readline().split())

while len(input_keyword) != 1:
    print('input one keyword: ', end='')
    input_keyword = list(stdin.readline().split())
input_keyword = input_keyword[0]
# print(important_words_by_year[2016])
for year in sorted(list(important_words_by_year.keys())):
    if input_keyword in important_words_by_year[year]:
        keyword_trend_dict[year] = important_words_by_year[year][input_keyword]
    else:
        keyword_trend_dict[year] = 0

years = list(keyword_trend_dict.keys())
values = list(keyword_trend_dict.values())

plt.figure(figsize=(10, 6))
plt.plot(years, values, marker='o', linestyle='-')

plt.title('Keyword Trend: {}'.format(input_keyword))
plt.xlabel('Year')
plt.ylabel('Value')
plt.grid(True)
plt.xticks(years, rotation=45)
plt.tight_layout()

plt.show()

