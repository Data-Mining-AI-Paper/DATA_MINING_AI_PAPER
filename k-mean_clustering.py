import pickle
import numpy as np
import parmap

from itertools import repeat
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


from tf_idf import MyTfidfVectorizer

def process_kmeans(X, k):
    initial_centers = kmeans_plusplus_initializer(X.todense(), k).initialize()
    kmeans_instance = kmeans(X.todense(), initial_centers)
    kmeans_instance.process()
    print(k)
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
    max_clusters=10
    print('Processing K-means clustering...')
    with Pool(cpu_count()) as pool:
        kmeans_instances = pool.starmap(process_kmeans, zip(repeat(X), range(2, max_clusters + 2)))
    print('Done')

    # Plot the elbow graph
    distortions = [ki.get_total_wce() for ki in kmeans_instances]
    plt.plot(range(2, max_clusters + 2), distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.savefig("output/k-means/Elbow Method.png")
    # plt.close()

    # Find the optimal k based on the elbow point
    # num_clusters = 4
    num_clusters = np.argmin(distortions) + 2
    print(f"The optimal number of clusters (k) is: {num_clusters}")

    with open("optimal_k-means_instance.pickle","wb") as f:
        pickle.dump(kmeans_instances[num_clusters-2], f)

    # Perform k-means clustering with the optimal k
    kmeans_clusters = kmeans_instances[num_clusters-2].get_clusters()

    # number of paper in cluster
    print('K-means cluster')
    for i in range(num_clusters):
        print(f"Cluster #{i+1}: {len(kmeans_clusters[i])}")


    svd = TruncatedSVD(n_components=3, random_state=42)
    Y_3d = svd.fit_transform(X)

    # 시각화
    # 3D Visualization with multiple views and zoomed-in versions
    fig = plt.figure(figsize=(30, 20))

    # List of (elev, azim) angles for different views
    view_angles = [(10,-30),(10,10),(20, 30),(20, 90), (30, 75)]  # Add more views if needed

    for idx, (elev, azim) in enumerate(view_angles, start=1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
    
        for i in range(num_clusters):
            cluster_points = [Y_3d[paper] for paper in kmeans_clusters[i]]
            cluster_points = np.array(cluster_points)
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"K-means Cluster {i + 2}")

        ax.set_title(f'3D K-means Clustering (View {idx})')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.legend()
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig("output/k-means/k-means_cluster_3d_multiple_views_zoomed.png")
 
