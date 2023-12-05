import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
from wordcloud import WordCloud
from collections import defaultdict

from tool import *
from preprocess import preprocess
from tf_idf import MyTfidfVectorizer

print("loading preprocessed data...", end='')
with open("preprocessed_ACL_PAPERS.pickle","rb") as f:
    data = pickle.load(f)
print("Done")

data_by_year =  defaultdict(list)
for paper in data:
    data_by_year[paper.year].append(paper)

important_words_by_year = {}
for year, one_year_data in tqdm(data_by_year.items()):
    abstracts = [paper.abstract for paper in one_year_data]

    vectorizer = MyTfidfVectorizer()
    X = vectorizer.fit_transform(abstracts)

    important_words = vectorizer.get_important_words(X, k=20, threshold=0.17)
    # print("important words of paper:", data[0].title)
    # print(*important_words[0].items(), sep='\n')

    important_words_by_year[year] = merge_dict_by_sum(important_words)
    
for year, year_tfidf in important_words_by_year.items():
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud = wordcloud.generate_from_frequencies(year_tfidf)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(str(year))  # 연도를 그래프 제목으로 추가
    plt.axis('off')
    # plt.show()
    plt.savefig(f'output/wordcloud/wordcloud_{year}.png')
    # plt.close()
