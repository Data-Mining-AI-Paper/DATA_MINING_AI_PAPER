import pickle
import numpy as np
import parmap

from itertools import repeat
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import TruncatedSVD
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from joblib import Parallel, delayed

from tf_idf import MyTfidfVectorizer

def process_kmeans(X, k):
    initial_centers = kmeans_plusplus_initializer(X.todense(), k).initialize()
    kmeans_instance = kmeans(X.todense(), initial_centers)
    kmeans_instance.process() 
    print(k)
    return kmeans_instance.get_total_wce()

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
    max_clusters=15
    print('Processing K-means clustering...')
    with Pool(cpu_count()) as pool:
        distortions = pool.starmap(process_kmeans, zip(repeat(X), range(1, max_clusters + 1)))
    
    print('Done')

    # Plot the elbow graph 
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.savefig("output/k-means/Elbow Method.png")
    # plt.close()

    # Find the optimal k based on the elbow point
    # num_clusters = 4
    num_clusters = np.argmin(distortions) + 1
    print(f"The optimal number of clusters (k) is: {num_clusters}")

   
