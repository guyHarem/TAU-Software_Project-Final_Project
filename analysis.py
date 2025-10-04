# Performs clustering analysis using K-means and SymNMF methods and compares their performance
import sys
import math
import pandas as pd
import numpy as np
import kmeans as KMeans
import symnmf as SymNMF
from sklearn.metrics import silhouette_score

    
def findClosestCentroid(vector, centroids):
    """Finds the index of the closest centroid to a given vector using Euclidean distance.
    
    Args:
        vector (numpy.ndarray): The vector to find its closest centroid.
        centroids (numpy.ndarray): Matrix of centroids.
        
    Returns:
        int: Index of the closest centroid based on minimum Euclidean distance.
    """
    vector = np.array(vector)  # Convert to numpy array
    centroids = np.array(centroids)  # Convert to numpy array
    
    # Vectorized distance calculation
    distances = np.linalg.norm(centroids - vector, axis=1)
    
    # Return index of minimum distance
    return np.argmin(distances)


def findKmeansLabels(vectors, k):
    """Performs K-means clustering and returns cluster labels.

    Args:
        vectors (pandas.DataFrame): Input data vectors.
        k (int): Number of clusters.

    Returns:
        list: Cluster labels (0 to k-1) for each input vector.
    """
    vectors = vectors.to_numpy()  # Convert DataFrame to numpy array
    kmeans_matrix = KMeans.doKmeans(vectors, k)
    kmeans_labels = [-1 for i in range(len(vectors))]
    for i in range(len(vectors)):
        kmeans_labels[i] = findClosestCentroid(vectors[i], kmeans_matrix)
    return kmeans_labels
        

def findSymNMFLabels(vectors, k):
    """Performs SymNMF clustering and returns cluster labels.

    Args:
        vectors (pandas.DataFrame): Input data vectors.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Cluster labels (0 to k-1) for each input vector based on highest association score.
    """
    vectors = vectors.values.tolist()  # Convert DataFrame to list of lists
    symnmf_matrix = SymNMF.doSymNMF(vectors, k)

    return np.array(symnmf_matrix).argmax(axis=1)


def main():
    """Main function for clustering analysis.

    Command line arguments:
        k (int): Number of clusters
        input_file (str): Path to input CSV file

    Prints:
        nmf: <symnmf_silhouette_score>
        kmeans: <kmeans_silhouette_score>

    Raises:
        Exception: If an error occurs during execution
    """
    try:
        input_data = sys.argv
        k, input_file = int(input_data[1]), input_data[2]

        vectors = pd.read_csv(input_file, header=None)  # Read the input file

        kmeans_labels = findKmeansLabels(vectors, k)  # Get K-means labels
        symnmf_labels = findSymNMFLabels(vectors, k)  # Get SymNMF labels

        # Calculate silhouette scores
        kmeans_silhouette = silhouette_score(vectors, kmeans_labels)
        symnmf_silhouette = silhouette_score(vectors, symnmf_labels)

        # Print results
        print(f"K-means silhouette score: {kmeans_silhouette:.4f}")
        print(f"SymNMF silhouette score: {symnmf_silhouette:.4f}")

    except Exception:
        print("An Error Has Occurred")
    
if __name__ == "__main__":
    main()


