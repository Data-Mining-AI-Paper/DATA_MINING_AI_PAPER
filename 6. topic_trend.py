import pickle

from collections import defaultdict
from matplotlib import pyplot as plt

from tf_idf import MyTfidfVectorizer
from tool import merge_dict_by_sum

with open("preprocessed_ACL_PAPERS.pickle","rb") as f:
        data = pickle.load(f)

data_by_year =  defaultdict(list)
for paper in data:
    data_by_year[paper.year].append(paper)
data_by_year = dict(sorted(data_by_year.items()))
years = data_by_year.keys()

abstracts = [paper.abstract for paper in data]
vectorizer = MyTfidfVectorizer()
X = vectorizer.fit_transform(abstracts)

with open(f"output/k-means/kmeans_instance/word2vec-embedding_kmeans_instance_opt_k36.pickle","rb") as f:
    kmeans_instance = pickle.load(f)

kmeans_clusters = kmeans_instance.get_clusters()
num_clusters = len(kmeans_clusters)

num_of_papers_by_cluster = list(map(len, kmeans_clusters))

num_of_cluster_by_year = defaultdict(lambda: defaultdict(int))
for i, cluster in enumerate(kmeans_clusters):
    for idx in cluster:
        num_of_cluster_by_year[data[idx].year][i] += 1
num_of_cluster_by_year = dict(sorted(num_of_cluster_by_year.items()))

raito_of_cluster = []
for i in range(num_clusters):
    raito_of_cluster.append([num_of_cluster_by_year[year][i]/len(data_by_year[year])*100 for year in num_of_cluster_by_year.keys()])

# 순서 변경
kmeans_clusters = [kmeans_clusters[i] for i, _ in sorted(enumerate(raito_of_cluster), key=lambda x: max(x[1]), reverse=True)]
raito_of_cluster.sort(key=lambda x: max(x), reverse=True)

X_by_cluster = [[] for _ in range(num_clusters)]
for i, x in enumerate(X):
    for j, c in enumerate(kmeans_clusters):
        if i in c:
            X_by_cluster[j].append(x)
            break

cluster_main_topic = []
for xc in X_by_cluster:
    topics = sorted(merge_dict_by_sum(vectorizer.get_important_words(xc)).items(), key=lambda x: x[1], reverse=True)[:3]
    topic = ' '.join([t[0] for t in topics])
    cluster_main_topic.append(topic)



# plt.title('number of paper of cluster by year')
# plt.xlabel('Year')
# plt.ylabel('Ratio of Topic')
# plt.grid(True)
# for i in range(num_clusters):
#     plt.plot(years, )
# plt.savefig(f"output/k-means/topic_trend_k{num_clusters}.png")
# print("Done")

fig, axs = plt.subplots(3, 2, figsize=(20, 15))

# 각 서브플롯에 데이터 플로팅
axs[0, 0].set_xlabel('Years')
axs[0, 0].set_ylabel('Percent')
for i, c in enumerate(raito_of_cluster[:num_clusters//6]):
    axs[0, 0].plot(years, c, label=cluster_main_topic[i])

axs[0, 1].set_xlabel('Years')
axs[0, 1].set_ylabel('Percent')
for i, c in enumerate(raito_of_cluster[num_clusters//6:num_clusters//6*2]):
    axs[0, 1].plot(years, c, label=cluster_main_topic[i+num_clusters//4])

axs[1, 0].set_xlabel('Years')
axs[1, 0].set_ylabel('Percent')
for i, c in enumerate(raito_of_cluster[num_clusters//6*2:num_clusters//6*3]):
    axs[1, 0].plot(years, c, label=cluster_main_topic[i+num_clusters//6*2])

axs[1, 1].set_xlabel('Years')
axs[1, 1].set_ylabel('Percent')
for i, c in enumerate(raito_of_cluster[num_clusters//6*3:num_clusters//6*4]):
    axs[1, 1].plot(years, c, label=cluster_main_topic[i+num_clusters//6*3])

axs[2, 0].set_xlabel('Years')
axs[2, 0].set_ylabel('Percent')
for i, c in enumerate(raito_of_cluster[num_clusters//6*4:num_clusters//6*5]):
    axs[2, 0].plot(years, c, label=cluster_main_topic[i+num_clusters//6*4])

axs[2, 1].set_xlabel('Years')
axs[2, 1].set_ylabel('Percent')
for i, c in enumerate(raito_of_cluster[num_clusters//6*5:]):
    axs[2, 1].plot(years, c, label=cluster_main_topic[i+num_clusters//6*5])

for ax in axs.flat:
    ax.legend()

plt.tight_layout()
plt.savefig(f"output/k-means/topic_trend_k{num_clusters}_sub.png")
