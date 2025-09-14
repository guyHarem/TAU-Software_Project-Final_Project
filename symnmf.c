#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAXLINE 1024 // Define max line length for the buffer size
#define EPSILON 1e-4 // Define a small epsilon value for convergence checks
#define MAX_ITER 300 // Define maximum number of iterations for convergence


// Global variables to hold the number of vectors and their dimension
static int N_c, vectordim_c;  ///WHAT DOES THAT MEANS????


/**
* Prints a matrix of doubles with dimensions rsize*csize
* @param matrix: the matrix to be printed
* @param rsize: the number of rows
* @param csize: the number of columns
* @return void
*/
void matrix_print(double** matrix, int rsize, int csize)
{
    int i, j;
    for(i = 0; i < rsize; i++) {
        for(j = 0; j < csize; j++) {
            printf("%.4f ", matrix[i][j]);
            if(j != csize - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}


/**
 * frees a matrix of doubles with dimensions rsize*csize
 * @param matrix: the matrix to be freed
 * @param rsize: the number of rows
 * @return void
 */
void matrix_free(double** matrix, int rsize)
{
    for(int i = 0; i < rsize; i++) {
        free(matrix[i]);
    }
    free(matrix);
}




/**
 * Allocates memory for a matrix of doubles with dimensions rsize*csize
 * @param rsize: the number of rows
 * @param csize: the number of columns
 * @return Pointer to the allocated 2D matrix, or NULL if allocation fails
 */
double** matrix_malloc(int rsize, int csize)
{
    int i;
    double** matrix;
    
    /* Validate input parameters */
    if (rsize <= 0 || csize <= 0) {
        fprintf(stderr, "An Error Has Occured", rsize, csize);
        return NULL;
    }
    
    /* Allocate memory for row pointers */
    matrix = (double**)malloc(rsize * sizeof(double*));
    if (matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }
    
    /* Allocate memory for each row */
    for (i = 0; i < rsize; i++) {
        matrix[i] = (double*)malloc(csize * sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "An Error Has Occured", i);
            /* Free previously allocated rows */
            matrix_free(matrix, i);
            return NULL;
        }
    }
    return matrix;
}



/**
 * Multiplies two matrices: matrix_a (a_rsize x a_csize) and matrix_b (a_csize x b_csize)
 * @param matrix_a: first input matrix
 * @param matrix_b: second input matrix
 * @param a_rsize: number of rows in matrix_a
 * @param a_csize: number of columns in matrix_a (and rows in matrix_b)
 * @param b_csize: number of columns in matrix_b
 * @return The allocated matrix, or NULL on failure
 */
double** matrix_multiply(double** matrix_a, double** matrix_b, int a_rsize, int a_csize, int b_csize) {
    int i, j, k;
    double** matrix_r = matrix_malloc(a_rsize, b_csize);
    if(matrix_r == NULL) {
        fprintf(stderr, "An Error Has Occured"); // memory allocation failed
        return NULL;
    }
    
    for(i = 0; i < a_rsize; i++) {
        for(j = 0; j < b_csize; j++) {
            matrix_r[i][j] = 0.0;
            for(k = 0; k < a_csize; k++) {
                matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
    return matrix_r;
}


/**
 * Transposes a matrix of doubles with dimensions rsize*csize
 * @param matrix: the matrix to be transposed
 * @param rsize: the number of rows
 * @param csize: the number of columns
 * @return The allocated transposed matrix, or NULL on failure
 */
double** matrix_transpose(double** matrix, int rsize, int csize) {
    int i, j;
    double** matrix_t = matrix_malloc(csize, rsize);
    if(matrix_t == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    for(i = 0; i < rsize; i++) {
        for(j = 0; j < csize; j++) {
            matrix_t[j][i] = matrix[i][j];
        }
    }
    return matrix_t;
}

/**
 * Subtracts two matrices: matrix_a and matrix_b, both of dimensions rsize*csize
 * @param matrix_a: first input matrix
 * @param matrix_b: second input matrix
 * @param rsize: number of rows in both matrices
 * @param csize: number of columns in both matrices
 * @return The allocated matrix, or NULL on failure
 */
double** matrix_subtract(double** matrix_a, double** matrix_b, int rsize, int csize) {
    int i, j;
    double** matrix_r = matrix_malloc(rsize, csize);
    if(matrix_r == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    for(i = 0; i < rsize; i++) {
        for(j = 0; j < csize; j++) {
            matrix_r[i][j] = matrix_a[i][j] - matrix_b[i][j];
        }
    }
    return matrix_r;
}


/**
 * Computes the Euclidean distance between two vectors of given dimension
 * @param vec1: first input vector
 * @param vec2: second input vector
 * @param vecdim: dimension of both vectors
 * @param is_squared: if non-zero, returns squared distance; otherwise returns actual distance
 * @return The Euclidean distance as a double
 */
double euc_dist(double* vec1, double* vec2, int vecdim, int is_squared) { //USE VECDIM_C INSTEAD OF PARAMETER
    double sum = 0.0;
    for(int i = 0; i < vecdim; i++) {
        double diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return is_squared ? sum : sqrt(sum); //WHY WE NEED THIS???
}



/**
 * Creates a similarity matrix from data vectors using Gaussian kernel
 * If vectors are similar → high similarity value (close to 1)
 * If vectors are different → low similarity value (close to 0)
 * @param matrix: input data matrix (N_c x vectordim_c)
 * @param rsize: number of data points (N_c)
 * @param csize: dimension of each data point (vectordim_c)
 * @return The allocated similarity matrix, or NULL on failure
 */
double** matrix_sym(double** matrix, int rsize, int csize) //USE N_C AND VECDIM_C INSTEAD OF PARAMETERS
{
    int i, j;
    double value;
    double** sym_matrix = NULL;

    /* Allocate memory for the similarity matrix (rsize x rsize) */
    sym_matrix = matrix_malloc(rsize, rsize);
    if (sym_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    /* Calculate the similarity matrix */
    for (i = 0; i < rsize; i++) {
        for (j = i; j < rsize; j++) {
            if (i == j) {
                sym_matrix[i][j] = 0;  /* Diagonal elements set to 0 */
            }
            else {
                value = euc_dist(matrix[i], matrix[j], csize, 1);  /* Squared Euclidean distance */
                value = exp(-value / 2);  /* Gaussian similarity */
                
                sym_matrix[i][j] = value;
                sym_matrix[j][i] = value;  /* Matrix is symmetric */
            }
        }
    }

    return sym_matrix;
}


/**
 * Computes the Degree Diagonal Matrix (DDG) for a given matrix //MAKE SURE TO USE MATRIX_SYM BEFORE USING THIS FUNCTION
 * @param matrix: input matrix
 * @param rsize: number of rows in the matrix
 * @param csize: number of columns in the matrix
 * @return The allocated DDG matrix, or NULL on failure
 */
double** matrix_ddg(double** matrix, int rsize, int csize) {
    int i, j;
    double sum;
    double** ddg_matrix = matrix_malloc(rsize, rsize);
    if(ddg_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    for(i = 0; i < rsize; i++) {
        sum = 0.0;
        for(j = 0; j < csize; j++) {
            sum += matrix[i][j];
        }
        for(j = 0; j < rsize; j++) {
            ddg_matrix[i][j] = (i == j) ? sum : 0.0; // Diagonal matrix
        }
    }
    return ddg_matrix;
}


/**
 * Creates D^(-1/2) matrix from diagonal degree matrix
 * @param D: diagonal degree matrix
 * @param size: matrix size
 * @return D^(-1/2) matrix, or NULL on failure
 */
double** matrix_inv_sqrt(double** D, int size) {
    int i, j;
    double** D_inv_sqrt = matrix_malloc(size, size);
    if (D_inv_sqrt == NULL) return NULL;
    
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            if (i == j && D[i][i] > 0) {
                D_inv_sqrt[i][j] = 1.0 / sqrt(D[i][i]);
            } else {
                D_inv_sqrt[i][j] = 0.0;
            }
        }
    }
    return D_inv_sqrt;
}

/**
 * Computes the normalized similarity matrix using SymNMF approach
 * Given original data matrix, computes: W = D^(-1/2) * A * D^(-1/2)
 * @param matrix: input data matrix (N x d)
 * @param rsize: number of data points (N)
 * @param csize: dimension of each data point (d)
 * @return The normalized similarity matrix W, or NULL on failure
 */
double** matrix_norm(double** matrix, int rsize, int csize) {
    double** A = matrix_sym(matrix, rsize, csize);
    if (A == NULL) return NULL;
    
    double** D = matrix_ddg(A, rsize, rsize);
    if (D == NULL) {
        matrix_free(A, rsize);
        return NULL;
    }

    double** D_inv_sqrt = matrix_inv_sqrt(D, rsize);
    if (D_inv_sqrt == NULL) {
        matrix_free(A, rsize);
        matrix_free(D, rsize);
        return NULL;
    }
    
    double** temp = matrix_multiply(D_inv_sqrt, A, rsize, rsize, rsize);
    double** W = NULL;
    if (temp != NULL) {
        W = matrix_multiply(temp, D_inv_sqrt, rsize, rsize, rsize);
        matrix_free(temp, rsize);
    }
    
    // Clean up all intermediate matrices after computations
    matrix_free(A, rsize);
    matrix_free(D, rsize);
    matrix_free(D_inv_sqrt, rsize);
    return W;
}

/**
 * Computes the Frobenius norm (matrix 2-norm) of a matrix
 * @param matrix: input matrix
 * @param rsize: number of rows
 * @param csize: number of columns
 * @param is_squared: if non-zero, returns squared Frobenius norm; otherwise returns actual Frobenius norm
 * @return The Frobenius norm as a double
 */
double frobenius_norm(double** matrix, int rsize, int csize, int is_squared) 
{
    double sum = 0.0;
    for(int i = 0; i < rsize; i++) {
        for(int j = 0; j < csize; j++) {
            sum += matrix[i][j] * matrix[i][j];
        }
    }
    return is_squared ? sum : sqrt(sum);
}


/**
 * Checks convergence between two matrices based on relative Frobenius norm difference
 * @param matrix_a: first input matrix
 * @param matrix_b: second input matrix
 * @param rsize: number of rows in both matrices
 * @param csize: number of columns in both matrices
 * @param EPSILON: convergence threshold
 * @return 1 if converged, 0 otherwise
 */
double matrix_convergence(double** matrix_a, double** matrix_b, int rsize, int csize)
{
    double norm_diff = frobenius_norm(matrix_subtract(matrix_a, matrix_b, rsize, csize), rsize, csize, 0);
    double norm_a = frobenius_norm(matrix_a, rsize, csize, 0);
    if (norm_a == 0) {
        return norm_diff < EPSILON; // Avoid division by zero
    }
    return (norm_diff / norm_a) < EPSILON;
}

// double matrix_convergence(double** matrix1, double** matrix2, int n, int m)
// {
//     double** result_matrix = matrix_substraction(matrix1, matrix2, n, m);
//     double norm = forbius_norm(result_matrix, n, m, 1);
//     matrix_free(result_matrix, n);
    
//     return norm;
// }





/**
 * Computes the average of all elements in a matrix
 * @param matrix: input data matrix
 * @param rsize: number of rows in input matrix
 * @param csize: number of columns in input matrix
 * @return The average value of all elements in the matrix
 */
double matrix_avg(double** matrix, int rsize, int csize) {
    double sum = 0.0;
    for(int i = 0; i < rsize; i++) {
        for(int j = 0; j < csize; j++) {
            sum += matrix[i][j];
        }
    }
    return sum / (rsize * csize);
}



/**
 * Initializes matrix H with random values from interval [0, 2*sqrt(avg/k)]
 * @param matrix: input data matrix to compute H from
 * @param rsize: number of rows in input matrix
 * @param csize: number of columns in input matrix
 * @param k: number of clusters/factors //IS THIS THE SAME K AS IN THE COMMAND LINE???
 * @return The initialized H matrix, or NULL on failure
 */
double** matrix_init_H(double** matrix, int rsize, int csize, int k) {
    int i, j;
    double avg, upper_bound;
    double** H = NULL;
    
    // Allocate memory for H matrix (N x k)
    H = matrix_malloc(rsize, k);
    if (H == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }
    
    // Calculate average using matrix_avg function
    avg = matrix_avg(matrix, rsize, csize);
    
    // Calculate upper bound for random values
    upper_bound = 2.0 * sqrt(avg / k);
    
    // Seed random number generator (call once) //DO IT IN THE PYTHON PART???
    srand(time(NULL));
    
    // Populate H with random values in [0, upper_bound]
    for (i = 0; i < rsize; i++) {
        for (j = 0; j < k; j++) {
            H[i][j] = ((double)rand() / RAND_MAX) * upper_bound;
        }
    }
    return H;
}




// MISSING SOME CODE OF UPDATE H FUNCTION AND SYMNMF FUNCTION





char* dup_string(char* str) //WE WERE NOT ALLWOED TO USE strpcpy?
 {
    char* str_copy = malloc(strlen(str) + 1);
    if(str_copy == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }
        strcpy(str_copy, str);
    return str_copy;
}


double** read_vectors_from_file(const char* filename, int* out_rsize, int* out_csize) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    } 
    char line[MAXLINE];
    int rsize = 0;
    int csize = -1;
    double** matrix = NULL;
    while (fgets(line, sizeof(line), file)) {
        char* token;                                        
        int col = 0;
        token = strtok(line, " ");
        while (token != NULL) {
            double value = atof(token);
            if (csize == -1) {
                csize = 1;
            } else {
                csize++;
            }
            token = strtok(NULL, " ");
        }
        rsize++;
    }
    fclose(file);
    *out_rsize = rsize;
    *out_csize = csize;
    return matrix;