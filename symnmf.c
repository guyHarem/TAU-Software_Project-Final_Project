#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAXLINE 1024 /* Define max line length for the buffer size */
#define EPSILON 1e-4 /* Define a small epsilon value for convergence checks */
#define MAX_ITER 300 /* Define maximum number of iterations for convergence */
#define BETA 0.5 /* Define beta parameter for update rule */


/* Global variables to hold the number of vectors and their dimension that were readed from file */
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
    int i;
    for(i = 0; i < rsize; i++) {
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
        fprintf(stderr, "An Error Has Occured");
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
            fprintf(stderr, "An Error Has Occured");
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
        fprintf(stderr, "An Error Has Occured"); /* memory allocation failed */
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
 * @return The Euclidean distance as a double
 */
double euc_dist(double* vec1, double* vec2, int vecdim) {
    int i;
    double sum = 0.0;
    for(i = 0; i < vecdim; i++) {
        double diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return sum;
}


/**
 * Creates a similarity matrix from input vectors using Gaussian kernel
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
                value = euc_dist(matrix[i], matrix[j], csize);  /* Squared Euclidean distance */
                value = exp(-value / 2);  /* Gaussian similarity */
                
                sym_matrix[i][j] = value;
                sym_matrix[j][i] = value;  /* Matrix is symmetric */
            }
        }
    }

    return sym_matrix;
}


/**
 * Computes the Degree Diagonal Matrix (DDG) for a given matrrix
 * @param matrix: input matrix
 * @param rsize: number of rows in the matrix
 * @param csize: number of columns in the matrix
 * @return The allocated DDG matrix, or NULL on failure
 */
double** matrix_ddg(double** matrix, int rsize, int csize) {

    int i, j;
    double sum;
    double** sym_matrix = NULL;
    double** ddg_matrix = NULL;

    sym_matrix = matrix_sym(matrix, rsize, csize); /* calculating similarity matrix first */
    if(sym_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    ddg_matrix = matrix_malloc(rsize, rsize);
    if(ddg_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    for(i = 0; i < rsize; i++) {
        sum = 0.0;
        for(j = 0; j < rsize; j++) {
            sum += sym_matrix[i][j]; /* Compute row sum */
        }
        for(j = 0; j < rsize; j++) {
            ddg_matrix[i][j] = (i == j) ? sum : 0.0; /* Diagonal matrix */
        }
    }

    matrix_free(sym_matrix, rsize); /* Free the intermediate similarity matrix */
    return ddg_matrix;
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

    int i, j;
    double** A = NULL;
    double** D = NULL;
    double** W = NULL;

    A = matrix_sym(matrix, rsize, csize);
    if(A == NULL) return NULL;

    D = matrix_ddg(matrix, rsize, csize);
    if(D == NULL) {
        matrix_free(A, rsize);
        return NULL;
    }

    W = matrix_malloc(rsize, rsize);
    if(W == NULL) {
        matrix_free(A, rsize);
        matrix_free(D, rsize);
        return NULL;
    }

    for(i = 0; i < rsize; i++) {
        for(j = 0; j < rsize; j++) {
            W[i][j] = A[i][j] / sqrt(D[i][i] * D[j][j]); /* Normalization step */
        }
    }

    /* Clean up all intermediate matrices after computations */
    matrix_free(A, rsize);
    matrix_free(D, rsize);

    return W;
}



/**
 * Computes the Frobenius norm (matrix 2-norm) of a matrix
 * @param matrix: input matrix
 * @param rsize: number of rows
 * @param csize: number of columns
 * @return The Frobenius norm as a double
 */
double frobenius_norm(double** matrix, int rsize, int csize) 
{
    int i, j;
    double sum = 0.0;
    for(i = 0; i < rsize; i++) {
        for(j = 0; j < csize; j++) {
            sum += matrix[i][j] * matrix[i][j];
        }
    }
    return sum;
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
    double** diff_matrix = NULL;
    double norm_diff = 0.0;

    diff_matrix = matrix_subtract(matrix_a, matrix_b, rsize, csize);
    if (diff_matrix == NULL) {
        return 0; /* Error case - assume not converged */
    }

    norm_diff = frobenius_norm(diff_matrix, rsize, csize);
    matrix_free(diff_matrix, rsize);

    return norm_diff < EPSILON; /* Return 1 if converged, 0 otherwise */
}


/** Updates the H matrix using the SymNMF multiplicative update rule
 * H = H * (1 - β + β * (WH) / (H(H^T*H)))
 * @param W: input symmetric normalized similarity matrix (N x N)
 * @param H: current factor matrix (N x k)
 * @param N: number of rows/columns in W and rows in H
 * @param k: number of columns in H (number of clusters)
 * @return The updated H matrix, or NULL on failure
 **/
double** update_H(double** W, double** H, int N, int k)
{
    int i, j;
    double** H_new = NULL;
    double** num_matrix = NULL;
    double** Ht = NULL;
    double** HHt = NULL;
    double** denom_matrix = NULL;

    /* Allocate memory for the new H matrix */
    H_new = matrix_malloc(N, k);
    if(H_new == NULL)
        return NULL; /* Memory allocation failed */
    
    /* Compute numerator matrix: W * H */
    num_matrix = matrix_multiply(W, H, N, N, k);
    if(num_matrix == NULL) {
        matrix_free(H_new, N);
        return NULL; /* Memory allocation failed */
    }

    /* Compute H transpose for denominator calculation */
    Ht = matrix_transpose(H, N, k);
    if(Ht == NULL) {
        matrix_free(H_new, N);
        matrix_free(num_matrix, N);
        return NULL; /* Memory allocation failed */
    }

    /* Compute H * H^T */
    HHt = matrix_multiply(H, Ht, N, k, N);
    if(HHt == NULL) {
        matrix_free(H_new, N);
        matrix_free(num_matrix, N);
        matrix_free(Ht, k);
        return NULL; /* Memory allocation failed */
    }

    /* Compute denominator matrix: H * (H^T * H) */
    denom_matrix = matrix_multiply(HHt, H, N, N, k);
    if (denom_matrix == NULL) {
        matrix_free(H_new, N);
        matrix_free(num_matrix, N);
        matrix_free(Ht, k);
        matrix_free(HHt, k);
        return NULL; /* Memory allocation failed */
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
    matrix_free(HHt, k);
    matrix_free(denom_matrix, N);

    return H_new;
}

/** 
 * Advances the previous H matrix to the new H matrix for the next iteration
 * @param H_prev: previous H matrix (to be updated)
 * @param H_new: new H matrix (source of updated values)
 * @param rsize: number of rows in H
 * @param csize: number of columns in H
*/
void advance_H(double** H_prev, double** H_new, int rsize, int csize)
{
    int i,j;
    for(i=0;i<rsize;i++)
    {
        for(j=0;j<csize;j++)
        {
            H_prev[i][j] = H_new[i][j]; /* Copy new values to previous matrix */
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
            return NULL; /* Error during update */
        }

        if (matrix_convergence(H_new, H, N, k)) {
            matrix_free(H, N);  // Free H^(t)
            return H_new;       // Return H^(t+1) - this is correct!
        }
        
        advance_H(H, H_new, N, k);
        matrix_free(H_new, N);
        H_new = NULL;
    }

    // If max iterations reached without convergence
    return H;  // Return H^(t)
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
double** read_vectors_from_file(const char *filename)
{
    char line[MAXLINE];
    char* token;
    int i,j;
    int row_count = 0;
    int col_count = 0;
    double** matrix = NULL;

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    /* First pass to determine the number of rows and columns */
    while (fgets(line, sizeof(line), file)) {
        row_count++;

        /* Count columns in the first row */
        if (row_count == 1) {
            char* temp = duplicateString(line);  /* Duplicate line for counting columns */
            char* token = strtok(temp, ",");
            while (token != NULL) {
                col_count++;
                token = strtok(NULL, ",");
            }
            free(temp);
        }
    }
    N_const = row_count;
    vectordim_const = col_count;

    /* Allocate memory for the 2D matrix */
    matrix = matrix_malloc(N_const, vectordim_const);

    /* Reset file pointer to beginning and read values into matrix */
    rewind(file);
    i = 0;
    while (fgets(line, sizeof(line), file)) {
        j = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            matrix[i][j++] = atof(token);  /* Convert token to double and store in matrix */
            token = strtok(NULL, ",");
        }
        i++;
    }

    fclose(file);
   
    return matrix;
}


int main(int argc, char* argv[])
{
    char* goal;
    double** data_matrix = NULL;
    double** goal_matrix = NULL;
    if(argc != 3) 
    {
        fprintf(stderr, "An Error Has Occured");
        return 1;
    }
    goal = duplicateString(argv[1]);
    data_matrix = read_vectors_from_file(argv[2]);

    (void)argc;

    if(data_matrix == NULL || goal == NULL) 
    {
        free(goal);
        matrix_free(data_matrix, N_const);
        fprintf(stderr,"An Error Has Occured");
        return 1;
    }

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
    matrix_print(goal_matrix, N_const, N_const);
    matrix_free(goal_matrix, N_const);
    matrix_free(data_matrix, N_const);
    free(goal);
    return 0;
}
