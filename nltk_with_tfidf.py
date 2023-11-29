from sys import stdin
from collections import defaultdict
import json
import re
from tokenizers import Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pyclustering.cluster.agglomerative import agglomerative
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.cure import cure

from wordcloud import WordCloud

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from tqdm import tqdm

import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')



def replace_non_alphabetic(sentence):
    # 정규 표현식을 사용하여 알파벳을 제외한 모든 문자를 공백으로 대체
    result = re.sub(r'[^a-zA-Z]', ' ', sentence)
    return result



a_json = open('./ACL_PAPERS.json', encoding = 'utf-8')
a_dict = json.load(a_json)	#=> 파이썬 자료형(딕셔너리나 리스트)으로 반환
# test_json = json.dumps(a_dict, ensure_ascii=False, indent=2)
# print(a_dict[0]['abstract'])
print('-' * 20)

# input comes from STDIN (standard input)
# for line in a_dict[0]['abstract']:
    # remove leading and trailing whitespace
    # line = line.strip()
    # split the line into words

abstracts = [a_dict[i]['abstract'] for i in range(100)]
# abstracts.append(a_dict[293]['abstract'])

# 중요한 단어들을 담을 리스트
important_words_list = []

for abstract in tqdm(abstracts):
    # 텍스트 전처리: 소문자로 변환하고 문장부호 및 불용어 제거
    abstract = abstract.lower()
    for punctuation in string.punctuation:
        abstract = abstract.replace(punctuation, "")
    tokens = word_tokenize(abstract)
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # TF-IDF 계산
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([' '.join(tokens)])

    # 단어별 중요도 출력
    feature_names = tfidf.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    word_scores = list(zip(feature_names, scores))

    # 중요도 순으로 정렬하여 상위 단어 저장
    word_scores.sort(key=lambda x: x[1], reverse=True)
    threshold = 0.2  # 임계치 설정
    important_words = [[word, score] for word, score in word_scores if score > threshold]
    
    # 중요한 단어들을 리스트에 추가
    important_words_list.append(important_words)

# 결과 출력
for i, words in enumerate(important_words_list, 1):
    print(f"Abstract {i}의 중요한 단어들 및 중요도 수치:")
    for word, score in words:
        print(f"{word}: {score}")
    print()



# 클러스터 개수 설정
num_clusters = 4  # 바꿀 수 있는 클러스터 개수

# TF-IDF로 추출된 중요한 단어들을 이용하여 SVD로 차원 축소
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform([' '.join([word[0] for word in words]) for words in important_words_list])

svd = TruncatedSVD(n_components=2, random_state=42)
transformed = svd.fit_transform(tfidf_matrix)

# K-means 클러스터링 (pyclustering의 kmeans 사용)
print('--- start K-means clustering ---')
initial_centers = kmeans_plusplus_initializer(tfidf_matrix.todense(), num_clusters).initialize()
kmeans_instance = kmeans(tfidf_matrix.todense(), initial_centers)
kmeans_instance.process()
kmeans_clusters = kmeans_instance.get_clusters()

# CURE 클러스터링
print('--- start CURE clustering ---')
cure_instance = cure(data=tfidf_matrix.todense(), number_cluster=num_clusters, compression=0.3, number_represent_points=num_clusters)
cure_instance.process()
cure_clusters = cure_instance.get_clusters()

# 클러스터링 개수 확인 
print('K-means cluster')
for cluster_idx in range(num_clusters):
    print(f"{cluster_idx+1}: {len(kmeans_clusters[cluster_idx])}")
print('CURE clsuter')
for cluster_idx in range(num_clusters):    
    print(f"{cluster_idx+1}: {len(cure_clusters[cluster_idx])}")

# 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
for cluster in range(num_clusters):
    cluster_points = [transformed[i] for i in kmeans_clusters[cluster]]
    cluster_points = list(zip(*cluster_points))
    plt.scatter(cluster_points[0], cluster_points[1], label=f"K-means Cluster {cluster+1}")
plt.title('K-means Clustering')
plt.legend()

plt.subplot(1, 3, 2)
for cluster in range(num_clusters):
    cluster_points = [transformed[i] for i in cure_clusters[cluster]]
    cluster_points = list(zip(*cluster_points))
    plt.scatter(cluster_points[0], cluster_points[1], label=f"CURE Cluster {cluster+1}")
plt.title('CURE Clustering')
plt.legend()

plt.tight_layout()
plt.show()


# 클러스터 개수 설정
# num_clusters = 4  # 바꿀 수 있는 클러스터 개수

# # TF-IDF로 추출된 중요한 단어들을 이용하여 SVD로 차원 축소
# tfidf = TfidfVectorizer()
# tfidf_matrix = tfidf.fit_transform([' '.join([word[0] for word in words]) for words in important_words_list])

# svd = TruncatedSVD(n_components=2, random_state=42)
# transformed = svd.fit_transform(tfidf_matrix)

# # K-means 클러스터링
# kmeans = KMeans(n_clusters=num_clusters, random_state=42, init='k-means++')
# kmeans.fit(tfidf_matrix)

# # 클러스터링 결과를 시각화
# cmap = plt.cm.get_cmap('tab10', num_clusters)  # num_clusters에 맞게 색 생성
# colors = cmap.colors
# for i in range(len(transformed)):
#     plt.scatter(transformed[i][0], transformed[i][1], color=colors[kmeans.labels_[i]], marker='o')

# plt.title('Clustering Visualization')
# plt.show()


# 워드 클라우드
tmp_dict = dict()
for sub in important_words_list:
    for word, score in sub:
        tmp_dict[word] = score

print(tmp_dict)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tmp_dict)

# 워드 클라우드 시각화
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()





# tfidf = TfidfVectorizer()

# # 텍스트 전처리: 소문자로 변환하고 문장부호 및 불용어 제거
# preprocessed_abstracts = []
# for abstract in tqdm(abstracts):
#     abstract = abstract.lower()
#     for punctuation in string.punctuation:
#         abstract = abstract.replace(punctuation, "")
#     tokens = word_tokenize(abstract)
#     tokens = [word for word in tokens if word not in stopwords.words('english')]
#     preprocessed_abstracts.append(' '.join(tokens))

# # TF-IDF 계산
# tfidf_matrix = tfidf.fit_transform(preprocessed_abstracts)

# # K-means 클러스터링
# num_clusters = 5  # 클러스터 개수 설정
# kmeans = KMeans(n_clusters=num_clusters)
# kmeans.fit(tfidf_matrix)

# # 각 클러스터에 속한 문서들의 인덱스를 딕셔너리에 저장
# clusters = {}
# for i, label in enumerate(kmeans.labels_):
#     if label not in clusters:
#         clusters[label] = []
#     clusters[label].append(i)

# # 각 클러스터에 속한 문서들의 토큰 출력 (임계값 이상의 TF-IDF만 출력)
# threshold = 0.05
# for cluster, docs in clusters.items():
#     print(f"Cluster {cluster + 1}의 문서들:")
#     for doc_index in docs:
#         print(f"Abstract {doc_index + 1}:")
#         tokens = preprocessed_abstracts[doc_index].split()
#         tfidf_scores = tfidf_matrix[doc_index].toarray().flatten()
#         selected_tokens = [token for token, score in zip(tokens, tfidf_scores) if score > threshold]
#         print(selected_tokens)
#     print()





print('-' * 20)







words = a_dict[0]['abstract'].split()
word_dict = defaultdict(int)
    # increase counters
for word in words:
    word = replace_non_alphabetic(word)
    word_dict[word.lower()] += 1

word_list = sorted(word_dict.items(), key=lambda x: -x[1])

# for key, val in word_list:
#     print(key, val)