from collections import defaultdict
import pickle

from matplotlib import pyplot as plt

with open("preprocessed_ACL_PAPERS.pickle","rb") as f:
        data = pickle.load(f)

data_by_year =  defaultdict(list)
for paper in data:
    data_by_year[paper.year].append(paper)
data_by_year = dict(sorted(data_by_year.items()))
years = data_by_year.keys()


num_clusters = 29

# with open(f"kmeans_instance_k{num_clusters}.pickle","rb") as f:
with open(f"output/k-means/kmeans_instance/word2vec-embedding_kmeans_instance_k29.pickle","rb") as f:
# with open(f"output/k-means/kmeans_instance/kmeans_instance_k5.pickle","rb") as f:

    kmeans_instance = pickle.load(f)

kmeans_clusters = kmeans_instance.get_clusters()

num_of_cluster = list(map(len, kmeans_clusters))

num_of_cluster_by_year = defaultdict(lambda: defaultdict(int))
for i, cluster in enumerate(kmeans_clusters):
    for idx in cluster:
        num_of_cluster_by_year[data[idx].year][i] += 1
num_of_cluster_by_year = dict(sorted(num_of_cluster_by_year.items()))

raito_of_cluster = []
for i in range(num_clusters):
    raito_of_cluster.append([num_of_cluster_by_year[year][i]/len(data_by_year[year])*100 for year in num_of_cluster_by_year.keys()])
raito_of_cluster.sort(key=lambda x: max(x), reverse=True)
# plt.title('number of paper of cluster by year')
# plt.xlabel('Year')
# plt.ylabel('Ratio of Topic')
# plt.grid(True)
# for i in range(num_clusters):
#     plt.plot(years, )
# plt.savefig(f"output/k-means/topic_trend_k{num_clusters}.png")
# print("Done")

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# 각 서브플롯에 데이터 플로팅
axs[0, 0].set_title('Plot 1')
for i, c in enumerate(raito_of_cluster[:num_clusters//4]):
    axs[0, 0].plot(years, c, label=str(i))

axs[0, 1].set_title('Plot 2')
for i, c in enumerate(raito_of_cluster[num_clusters//4:num_clusters//4*2]):
    axs[0, 1].plot(years, c, label=str(i+num_clusters//4))

axs[1, 0].set_title('Plot 3')
for i, c in enumerate(raito_of_cluster[num_clusters//4*2:num_clusters//4*3]):
    axs[1, 0].plot(years, c, label=str(i+num_clusters//4*2))

axs[1, 1].set_title('Plot 4')
for i, c in enumerate(raito_of_cluster[num_clusters//4*3:]):
    axs[1, 1].plot(years, c, label=str(i+num_clusters//4*3))

for ax in axs.flat:
    ax.legend()

plt.tight_layout()
plt.savefig(f"output/k-means/topic_trend_k{num_clusters}_sub.png")
