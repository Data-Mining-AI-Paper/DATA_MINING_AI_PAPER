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


num_clusters = 11
with open(f"kmeans_instance_k{num_clusters}.pickle","rb") as f:
    kmeans_instance = pickle.load(f)

kmeans_clusters = kmeans_instance.get_clusters()

num_of_cluster = list(map(len, kmeans_clusters))

num_of_cluster_by_year = defaultdict(lambda: defaultdict(int))
for i, cluster in enumerate(kmeans_clusters):
    for idx in cluster:
        num_of_cluster_by_year[data[idx].year][i] += 1
num_of_cluster_by_year = dict(sorted(num_of_cluster_by_year.items()))


plt.title('number of paper of cluster by year')
plt.xlabel('Year')
plt.ylabel('number of paper')
plt.grid(True)
for i in range(num_clusters):
    plt.plot(years, [num_of_cluster_by_year[year][i]/len(data_by_year[year])*100 for year in num_of_cluster_by_year.keys()], marker='o', linestyle='-')
plt.savefig(f"output/k-means/topic_trend_k{num_clusters}.png")
print("Done")

