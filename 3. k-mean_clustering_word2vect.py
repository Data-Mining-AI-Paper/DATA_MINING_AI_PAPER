import pickle
from gensim.models.word2vec import Word2Vec

from itertools import repeat
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from sklearn.metrics import silhouette_score

from tf_idf import MyTfidfVectorizer

def process_kmeans(X, k): 
    initial_centers = kmeans_plusplus_initializer(X, k).initialize()
    kmeans_instance = kmeans(X, initial_centers)
    kmeans_instance.process()
    print(k)
    return kmeans_instance

def calculate_distortion(X,k):
    return process_kmeans(X,k).get_total_wce()


def calculate_silhouette(X, instances):
    silhouette_avg_list=[]
    for instance in instances:
        labels = np.concatenate([np.full(len(cluster), i) for i, cluster in enumerate(instance.get_clusters())])
        silhouette_avg = silhouette_score(X, labels)
        silhouette_avg_list.append(silhouette_avg)
    return silhouette_avg_list

if __name__ == "__main__":
    method_name = "word2vec-embedding"

    with open("preprocessed_ACL_PAPERS.pickle","rb") as f:
        data = pickle.load(f)

    abstracts = [paper.abstract for paper in data]

    parallel_num = cpu_count()
    sentences = [abstract.lower().split() for abstract in abstracts]
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=parallel_num)

    vectorizer = MyTfidfVectorizer()
    X = vectorizer.fit_transform(abstracts)

    X = vectorizer.embedding(X, model.wv)

    # K-means 클러스터링
    print('--- start K-means clustering ---')
    
    # Use multiprocessing to process kmean in parallel
    start=30
    end=41
    k_list = range(start, end)
    print('Processing K-means clustering...')
    with Pool(parallel_num) as pool:
        k_instances = pool.starmap(process_kmeans, zip(repeat(X), k_list))
    print('Done')
    
    distortions=[k.get_total_wce() for k in k_instances]
    
    silhouette_scores = calculate_silhouette(X,k_instances)

    # Plot the elbow graph
    plt.plot(k_list, distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.savefig(f"output/k-means/Elbow Method_{method_name}.png")
    
    # Plot the silhouette graph
    plt.figure()
    plt.plot(k_list, silhouette_scores, marker='o', label='Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.legend()
    plt.savefig(f"output/k-means/Silhouette Method_{method_name}.png")

    for i in range(start,end):
        with open(f"output/k-means/kmeans_instance/{method_name}_kmeans_instance_everything_{i}.pickle", "wb") as f:
            pickle.dump(k_instances[i-start],f)
    
    # Find the optimal k based on the elbow point
    num_clusters_elbow = k_list[np.argmin(distortions)]
    print(f"The optimal number of clusters (k) based on the elbow method is: {num_clusters_elbow}")

    # Find the optimal k based on the silhouette method
    num_clusters_silhouette = k_list[np.argmax(silhouette_scores)]
    print(f"The optimal number of clusters (k) based on the silhouette method is: {num_clusters_silhouette}")

    opt_num_clusters = int(input("Enter optimal cluster number:"))
    opt_kmeans_instance = k_instances[opt_num_clusters-start]
    with open(f"output/k-means/kmeans_instance/{method_name}_kmeans_instance_opt_k{opt_num_clusters}.pickle", "wb") as f:
        pickle.dump(opt_kmeans_instance, f)

    # number of paper in cluster
    opt_kmeans_clusters = opt_kmeans_instance.get_clusters()
    print('optimal K-means cluster')
    for i in range(opt_num_clusters):
        print(f"Cluster #{i+1}: {len(opt_kmeans_clusters[i])}")
