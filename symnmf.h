
/* C header file */
#ifndef SYMNMF_H
#define SYMNMF_H

void matrix_free(double **p, int n);
double** matrix_malloc(int n, int m);
double** matrix_sym(double** vectors, int N, int vecdim);
double** matrix_ddg(double** vectors, int N, int vecdim);
double** matrix_norm(double** vectors, int N, int vecdim);
double** matrix_symnmf(double** W, double** H, int N, int k);

#endif