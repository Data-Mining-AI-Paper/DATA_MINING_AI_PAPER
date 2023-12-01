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
    print(k)
    initial_centers = kmeans_plusplus_initializer(X.todense(), k).initialize()
    kmeans_instance = kmeans(X.todense(), initial_centers)
    kmeans_instance.process()
    return kmeans_instance

def calculate_distortion(X,k):
    return process_kmeans(X,k).get_total_wce()

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

    # K-means 클러스터링
    print('--- start K-means clustering ---')
    
    # Use multiprocessing to process kmean in parallel
    parallel_num = cpu_count()
    k_list = range(2, parallel_num + 2)
    print('Processing K-means clustering...')
    with Pool(parallel_num) as pool:
        distortions = pool.starmap(calculate_distortion, zip(repeat(X), k_list))
    print('Done')

    # Plot the elbow graph
    plt.plot(k_list, distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.savefig("output/k-means/Elbow Method.png")
    # plt.close()

    # Find the optimal k based on the elbow point
    num_clusters = k_list[np.argmin(distortions)]
    print(f"The optimal number of clusters (k) is: {num_clusters}")
    
    # Perform k-means clustering with the optimal k
    kmeans_instance = process_kmeans(X, num_clusters)
    with open(f"kmeans_instance_k{num_clusters}.pickle","wb") as f:
        pickle.dump(kmeans_instance, f)
    kmeans_clusters = kmeans_instance.get_clusters()

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
 
