import sys
import math
import pandas as pd
import numpy as np
import kmeans as KMeans
import symnmf as SymNMF
from sklearn.metrics import silhouette_score

    
def findClosestCentroid(vector, centroids):
    """Finds the index of the closest centroid to a given vector.
    
    Args:
        vector (list): The vector for which to find the closest centroid.
        centroids (list): List of current centroids.
        
    Returns:
        int: Index of the closest centroid.
    """
    vector = np.array(vector)  # Convert to numpy array
    centroids = np.array(centroids)  # Convert to numpy array
    
    # Vectorized distance calculation
    distances = np.linalg.norm(centroids - vector, axis=1)
    
    # Return index of minimum distance
    return np.argmin(distances)




def findKmeansLabels(vectors, k):
    """Finds K-means cluster labels for given vectors.

    Args:
        vectors (list): List of vectors to cluster.
        k (int): Number of clusters.

    Returns:
        list: Cluster labels for each vector.
    """
    vectors = np.array(vectors)
    kmeans_matrix = KMeans.doKmeans(vectors, k) 
    kmeans_labels = [-1 for i in range(len(vectors))]  # Initialize labels with -1
    
    for i in range(len(vectors)):
        kmeans_labels[i] = findClosestCentroid(vectors[i], kmeans_matrix)  # Assign label based on closest centroid
        
    return kmeans_labels



def findSymNMFLabels(vectors, k):
    """Finds SymNMF cluster labels for given vectors.

    Args:
        vectors (list): List of vectors to cluster.
        k (int): Number of clusters.

    Returns:
        list: Cluster labels for each vector.
    """
    vectors = np.array(vectors)
    symnmf_matrix = SymNMF.doSymNMF(vectors, k)  
    symnmf_labels = [-1 for i in range(len(vectors))]  # Initialize labels with -1

    for i in range(len(vectors)):
        symnmf_labels[i] = findClosestCentroid(vectors[i], symnmf_matrix)  # Assign label based on closest centroid

    return symnmf_labels


def main():
    try:
        input_data = sys.argv
        k, input_file = int(input_data[1]), input_data[2]

        vectors = pd.read_csv(input_file, header=None).values.tolist()  # Read the input file and convert it to a list of lists

        kmeans_labels = findKmeansLabels(vectors, k)  # Get K-means labels
        symnmf_labels = findSymNMFLabels(vectors, k)  # Get SymNMF labels

        # Calculate silhouette scores
        kmeans_silhouette = silhouette_score(vectors, kmeans_labels)
        symnmf_silhouette = silhouette_score(vectors, symnmf_labels)

        # Print results
        print(f"K-means silhouette score: {kmeans_silhouette:.4f}")
        print(f"SymNMF silhouette score: {symnmf_silhouette:.4f}")

    except Exception as e:
        print("An Error Has Occurred")
        return
    

if __name__ == "__main__":
    main()


