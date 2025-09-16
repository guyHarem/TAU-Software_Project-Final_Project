import math
import pandas as pd
import numpy as np


def assignClusters(vectors, centroids):
    """Assigns each vector to the closest centroid.
    
    Returns a list of lists where each list contains the vectors assigned 
    to the corresponding centroid.
    
    Args:
        vectors (list): List of vectors to be clustered.
        centroids (list): List of current centroids.
        
    Returns:
        list: Cluster assignment as list of lists of vectors (size: k*d).
    """
    clusters = [[] for i in range(len(centroids))] # Initialize clusters as a list of lists
    
    for vector in vectors:
        min_distance = math.inf
        cluster_index = 0
        for j in range(len(centroids)):
            distance = np.linalg.norm(vector - centroids[j])
            if distance < min_distance:
                min_distance = distance
                cluster_index = j
        clusters[cluster_index].append(vector)

    return clusters


def updateCentroids(centroids, clusters):
    """Updates the centroids by taking the mean of vectors in each cluster.
    
    Args:
        centroids (list): List of current centroids.
        clusters (list): List of lists where each list contains the vectors 
            assigned to the corresponding centroid.
            
    Returns:
        list: Updated centroids.
    """
    for i in range(len(centroids)):
        if clusters[i]: # Avoid division by zero
            centroids[i] = np.mean(clusters[i], axis=0)

    return centroids


def updateCentroidAndConvergence(centroids, clusters):
    """Checks if the centroids have converged and updates them.
    
    Updates centroids and determines if the algorithm has converged based on
    whether the change in centroid positions is below the threshold (0.0001).
    
    Args:
        centroids (list): List of current centroids.
        clusters (list): List of lists where each list contains the vectors 
            assigned to the corresponding centroid.
            
    Returns:
        bool: True if the centroids have converged, False otherwise.
    """
    convergence = True
    for i in range(len(centroids)):
        if clusters[i]:
            new_centroid = np.mean(clusters[i], axis=0)
        else:
            new_centroid = np.zeros(len(centroids[0]))

        if np.linalg.norm(new_centroid - centroids[i]) >= 0.0001:
            convergence = False
        centroids[i] = new_centroid
    return convergence


def doKmeans(vectors, k):
    """Performs k-means clustering on the given vectors.
    
    Implements the k-means clustering algorithm with a maximum of 300 iterations.
    Uses the first k vectors as initial centroids and iterates until convergence
    or maximum iterations are reached.
    
    Args:
        vectors (list): List of vectors to be clustered.
        k (int): Number of clusters to form.
        
    Returns:
        list: Final list of centroids after clustering.
    """
    centroids = vectors[:k].copy() # Choose first k vectors as initial centroids

    for i in range(300):
        # Assign each vector to the closest centroid
        clusters = assignClusters(vectors, centroids)

        # Update centroids
        if updateCentroidAndConvergence(centroids, clusters):
            break

    return centroids.tolist()