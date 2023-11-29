import pickle
import numpy as np
import parmap

from itertools import repeat
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import TruncatedSVD
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans

from tf_idf import MyTfidfVectorizer

def process_kmeans(X, k):
    initial_centers = kmeans_plusplus_initializer(X.todense(), k).initialize()
    kmeans_instance = kmeans(X.todense(), initial_centers)
    kmeans_instance.process()
    return kmeans_instance

if __name__ == "__main__":
    print("Loading preprocessed data...", end='')
    with open("preprocessed_ACL_PAPERS.pickle","rb") as f:
        data = pickle.load(f)
    print("Done")

    abstracts = [paper.abstract for paper in data]
    vectorizer = MyTfidfVectorizer()
    X = vectorizer.fit_transform(abstracts)
    important_words = vectorizer.get_important_words(X, k=20, threshold=0.17)


    vectorizer = MyTfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(paper.keys()) for paper in important_words])

    # # Standardize the data (important for K-means)
    # scaler = StandardScaler(with_mean=False) # Set with_mean to False for sparse matrices
    # tfidf_matrix_standardized = scaler.fit_transform(tfidf_matrix)

    # K-means 클러스터링
    print('--- start K-means clustering ---')
    
    # Use multiprocessing to process kmean in parallel
    max_clusters=8
    print('Processing K-means clustering...')
    with Pool(cpu_count()) as pool:
        kmeans_instances = pool.starmap(process_kmeans, zip(repeat(X), range(1, max_clusters + 1)))
    print('Done')

    # Plot the elbow graph
    distortions = [ki.get_total_wce() for ki in kmeans_instances]
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.savefig("Elbow Method.png")
    plt.close()

    # Find the optimal k based on the elbow point
    # num_clusters = 4
    num_clusters = np.argmin(distortions) + 1
    print(f"The optimal number of clusters (k) is: {num_clusters}")

    # Perform k-means clustering with the optimal k
    kmeans_clusters = kmeans_instances[num_clusters-1].get_clusters()

    # number of paper in cluster
    print('K-means cluster')
    for i in range(num_clusters):
        print(f"Cluster #{i+1}: {len(kmeans_clusters[i])}")


    svd = TruncatedSVD(n_components=2, random_state=42)
    Y = svd.fit_transform(X)

    # 시각화
    plt.figure(figsize=(10, 8))
    for i in range(num_clusters):
        cluster_points = [Y[paper] for paper in kmeans_clusters[i]]
        cluster_points = list(zip(*cluster_points))
        plt.scatter(cluster_points[0], cluster_points[1], label=f"K-means Cluster {i+1}")
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()
