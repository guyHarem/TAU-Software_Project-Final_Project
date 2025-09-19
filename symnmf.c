#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAXLINE 1024 // Define max line length for the buffer size
#define EPSILON 1e-4 // Define a small epsilon value for convergence checks
#define MAX_ITER 300 // Define maximum number of iterations for convergence
#define BETA 0.5 // Define beta parameter for update rule


// Global variables to hold the number of vectors and their dimension that were readed from file
static int N_const, vectordim_const; 


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
double euc_dist(double* vec1, double* vec2, int vecdim, int is_squared) {
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
 * @param matrix: input data matrix
 * @param rsize: number of data points
 * @param csize: dimension of each data point
 * @return The allocated similarity matrix, or NULL on failure
 */
double** matrix_sym(double** matrix, int rsize, int csize)
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
 * Checks convergence between two matrices based on Frobenius norm difference
 * Convergence rule: ||H^(t+1) - H^(t)||²_F < ε
 * @param matrix_a: current iteration matrix H^(t+1)
 * @param matrix_b: previous iteration matrix H^(t)
 * @param rsize: number of rows in both matrices
 * @param csize: number of columns in both matrices
 * @return 1 if converged (norm_diff < EPSILON), 0 otherwise
 */
int matrix_convergence(double** matrix_a, double** matrix_b, int rsize, int csize)
{
    double** diff_matrix = matrix_subtract(matrix_a, matrix_b, rsize, csize);
    if (diff_matrix == NULL) {
        return 0; // Error case - assume not converged
    }
    
    double norm_diff_squared = frobenius_norm(diff_matrix, rsize, csize, 1); // Use squared norm
    matrix_free(diff_matrix, rsize);
    
    return norm_diff_squared < EPSILON;
}




double** update_H(double** W, double** H, int N, int k)
{
    int i, j;
    /* Allocate memory for the new H matrix */
    double** H_new = matrix_malloc(N, k);
    /* Compute numerator matrix: W * H */
    double** num_matrix = matrix_multiply(W, H, N, N, k);
    /* Compute H transpose for denominator calculation */
    double** Ht = matrix_transpose(H, N, k);
    /* Compute H^T * H (conditional allocation to prevent cascading failures) */
    double** HtH = (Ht != NULL) ? matrix_multiply(Ht, H, k, N, k) : NULL;
    /* Compute denominator matrix: H * (H^T * H) */
    double** denom_matrix = (HtH != NULL) ? matrix_multiply(H, HtH, N, k, k) : NULL;
    
    /* Check if any allocation failed */
    if (H_new == NULL || num_matrix == NULL || Ht == NULL || HtH == NULL || denom_matrix == NULL) {
        /* Clean up any successfully allocated matrices */
        if (H_new) matrix_free(H_new, N);
        if (num_matrix) matrix_free(num_matrix, N);
        if (Ht) matrix_free(Ht, k);
        if (HtH) matrix_free(HtH, k);
        if (denom_matrix) matrix_free(denom_matrix, N);
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    /* Apply the SymNMF update formula: H_new = H * (1 - BETA + BETA * (num/denom)) */
    for (i = 0; i < N; i++) {
        for (j = 0; j < k; j++) {
            if (denom_matrix[i][j] != 0) { /* Avoid division by zero */
                /* Element-wise multiplication with update factor */
                H_new[i][j] = H[i][j] * (1 - BETA + BETA * (num_matrix[i][j] / denom_matrix[i][j]));
            } else {
                H_new[i][j] = H[i][j]; /* Keep original value if denominator is zero */
            }
        }
    }

    /* Free all intermediate matrices */
    matrix_free(num_matrix, N);
    matrix_free(Ht, k);
    matrix_free(HtH, k);
    matrix_free(denom_matrix, N);

    return H_new;
}

/** 
 * Advances the previous H matrix to the new H matrix for the next iteration
 * @param H_prev: previous H matrix (to be updated)
 * @param H_new: new H matrix (source of updated values)
 * @param N: number of rows in H
 * @param k: number of columns in H
*/
void advance_H(double** H_prev, double** H_new, int N, int k)
{
    int i,j;
    for(i=0;i<N;i++)
    {
        for(j=0;j<k;j++)
        {
            H_prev[i][j] = H_new[i][j];
        }
    }
}


/**
 * Performs Symmetric Non-negative Matrix Factorization (SymNMF) on matrix W
 * 
 * Iteratively updates H using the multiplicative rule: H = H * (1 - β + β * (WH) / (H(H^T*H)))
 * until convergence (||H_new - H||²_F < ε) or maximum iterations (MAX_ITER = 300) reached.
 * The algorithm factorizes W ≈ H * H^T where H is the non-negative factor matrix.
 * 
 * @param W: input symmetric normalized similarity matrix (N x N)
 * @param H: initial factor matrix (N x k) - modified in-place during iterations
 * @param N: number of rows/columns in W and rows in H
 * @param k: number of columns in H (number of clusters)
 * @return The final factor matrix H after convergence, or NULL on failure
 */
double** matrix_symnmf(double** W, double** H, int N, int k)
{
    int iter;
    double** H_new = NULL;

    for(iter = 0; iter < MAX_ITER; iter++) {
        H_new = update_H(W, H, N, k);
        if (H_new == NULL) {
            return NULL; // Error during update
        }

        if (matrix_convergence(H_new, H, N, k)) {
            break; // Converged
        }
        
        advance_H(H, H_new, N, k);
        matrix_free(H_new, N);
        H_new = NULL; // Reset for next iteration
    }
    return H;
}

/**
 * Duplicates a string by allocating new memory and copying the content
 * @param src: source string to duplicate
 * @return Pointer to the duplicated string, or NULL if allocation fails
 */
char* duplicateString(char* src)
{
    char* str;
    char* p;
    int len = 0;

    if(src == NULL)
    {
        printf("An error has occured");
        return NULL;
    }
    
    while (src[len])
        len++;
    str = malloc(len + 1);
    p = str;
    while (*src)
        *p++ = *src++;
    *p = '\0';
    return str;
}

/**
 * Reads vectors from a file and stores them in a dynamically allocated 2D array (matrix)
 * Each line in the file represents a vector, with elements separated by spaces.
 * The function also sets the global variables N_const and vectordim_const to the number of vectors
 * and their dimension, respectively.
 * @param filename: path to the input file
 * @return Pointer to the allocated 2D matrix of doubles, or NULL on failure
 */
double** read_vectors_from_file(const char* filename) {
    char line[MAXLINE];
    char* token;
    int rsize = 0;
    int csize = -1; // Initialize csize to -1 to detect first line
    int i, j;
    double** matrix = NULL;

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    // First pass: determine rsize and csize
    while (fgets(line, sizeof(line), file)) {
        int temp_csize = 0;
        token = strtok(line, " ");
        while (token != NULL) {
            temp_csize++;
            token = strtok(NULL, " ");
        }
        if (csize == -1) {
            csize = temp_csize; // Set csize based on the first line
        } else if (temp_csize != csize) {
            fprintf(stderr, "An Error Has Occured"); // Inconsistent column count
            fclose(file);
            return NULL;
        }
        rsize++;
    }

    // Allocate memory for the matrix
    matrix = matrix_malloc(rsize, csize);
    if (matrix == NULL) {
        fclose(file);
        return NULL;
    }

    // Second pass: read the actual data
    rewind(file);
    i = 0;
    while (fgets(line, sizeof(line), file) && i < rsize) {
        j = 0;
        token = strtok(line, " ");
        while (token != NULL && j < csize) {
            matrix[i][j] = atof(token);
            j++;
            token = strtok(NULL, " ");
        }
        i++;
    }

    fclose(file);
    
    // Set the static global variables
    N_const = rsize;
    vectordim_const = csize;
    
    return matrix;
}


int main(int argc, char* argv[])
{
    if(argc != 3) 
    {
        fprintf(stderr, "An Error Has Occured");
        return 1;
    }

    char* goal = duplicateString(argv[1]);
    double** data_matrix = read_vectors_from_file(argv[2]);

    (void)argc; // Unused parameter

    if(data_matrix == NULL || goal == NULL) 
    {
        free(goal);
        matrix_free(data_matrix, N_const);
        fprintf(stderr,"An Error Has Occured");
        return 1;
    }

    double** goal_matrix = NULL;

    if(!strcmp(goal,"sym"))
    {
        goal_matrix = matrix_sym(data_matrix, N_const, vectordim_const);
    }
    else if(!strcmp(goal,"ddg"))
    {
        goal_matrix = matrix_ddg(data_matrix, N_const, vectordim_const);
    }
    else if(!strcmp(goal,"norm"))
    {
        goal_matrix = matrix_norm(data_matrix, N_const, vectordim_const);
    }
    print_matrix(goal_matrix, N_const, N_const);
    matrix_free(goal_matrix, N_const);
    matrix_free(data_matrix, N_const);
    free(goal);
    return 0;
}