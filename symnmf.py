import sys
import math
import pandas as pd
import numpy as np
import mysymnmfsp as SymNMF

np.random.seed(1234)

def matrix_init_H(matrix, N, k):
    """Initialize the matrix H for NMF.

    Args:
        matrix (list of list of float): The input matrix to be used.
        N (int): The number of vectors (Rows) in the input matrix.
        k (int): The number of centroids (columns) in the factor matrix H.

    Returns:
        list of list of float: The initialized factor matrix H with shape (N, k).
    """
    mean = np.mean(matrix) # Calculate the average of all entries in the input matrix
    upper_bound = 2 * np.sqrt(mean / k) # Calculate the upper bound for the random values
    H = np.random.uniform(low=0, high=upper_bound, size=(N, k)) # Initialize H with random values from the interval [0, upper_bound]
    return H.tolist()


def doSymNMF(vectors, k):
    """Perform Symmetric Non-negative Matrix Factorization (SymNMF) on the input vectors.

    Args:
        vectors (list of list of float): The input vectors to be factorized.
        k (int): The number of clusters to form.

    Returns:
        list: A list of list of float representing the resulting matrix after performing SymNMF.
    """
    matrix_W = SymNMF.matrix_norm(vectors) # calling the norm function from symNMF module (C extension)
    matrix_H = matrix_init_H(matrix_W, len(vectors), k) # Initialize the factor matrix H
    matrix_goal = SymNMF.matrix_symnmf(matrix_W, matrix_H, k) # calling the symnmf function from symNMF module (C extension) to perform SymNMF
    return matrix_goal


def main():
    try:
        input_data = sys.argv
        k, goal, input_file = int(input_data[1]), input_data[2], input_data[3]

        vectors = pd.read_csv(input_file, header=None).values.tolist() # Read the input file and convert it to a list of lists

        match goal:
            case "sym":
                matrix_goal = SymNMF.matrix_sym(vectors) # Calling matrix_sym function in C to calculate the matrix
            case "ddg":
                matrix_goal = SymNMF.matrix_ddg(vectors) # Calling matrix_ddg function in C to calculate the matrix
            case "norm":
                matrix_goal = SymNMF.matrix_norm(vectors) # Calling matrix_norm function in C to calculate the matrix
            case "symnmf":
                matrix_goal = doSymNMF(vectors, k) # Calling doSymNMF function to perform SymNMF
            case _:
                print("An Error Has Occurred")
                return
                
        # print matrix_goal until 4 decimal points
        for row in matrix_goal:
            print(','.join(format(x, ".4f") for x in row)) ## WHY YOU NEED IT IF YOU DOT IT IN C?

    except Exception:
        print("An Error Has Occurred")
        
if __name__ == "__main__":
    main()