import pickle
from gensim.models.word2vec import Word2Vec

from itertools import repeat
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans

from tf_idf import MyTfidfVectorizer

def process_kmeans(X, k): 
    initial_centers = kmeans_plusplus_initializer(X, k).initialize()
    kmeans_instance = kmeans(X, initial_centers)
    kmeans_instance.process()
    print(k)
    return kmeans_instance

def calculate_distortion(X,k):
    return process_kmeans(X,k).get_total_wce()

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
    k_list = range(10, 111, 10)
    print('Processing K-means clustering...')
    with Pool(parallel_num) as pool:
        distortions = pool.starmap(calculate_distortion, zip(repeat(X), k_list))
    print('Done')

    # Plot the elbow graph
    plt.plot(k_list, distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.savefig(f"output/k-means/Elbow Method_{method_name}.png")

    # Find the optimal k based on the elbow point
    num_clusters = k_list[np.argmin(distortions)]
    print(f"The optimal number of clusters (k) is: {num_clusters}")
    
    # Perform k-means clustering with the optimal k
    kmeans_instance = process_kmeans(X, num_clusters)
    with open(f"output/k-means/kmeans_instance/{method_name}_kmeans_instance_k{num_clusters}.pickle","wb") as f:
        pickle.dump(kmeans_instance, f)
    kmeans_clusters = kmeans_instance.get_clusters()

    # number of paper in cluster
    print('K-means cluster')
    for i in range(num_clusters):
        print(f"Cluster #{i+1}: {len(kmeans_clusters[i])}")

