#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAXLINE 1024 // Define max line length for the buffer size


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









