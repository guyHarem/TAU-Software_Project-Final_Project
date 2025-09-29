#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>
#include "symnmf.h"


static int N, vectordim, k; ///WHAT DOES THAT MEANS????

/**
 * Converts a Python list of lists (2D list) to a C 2D array (matrix)
 * @param pylist: The input Python list of lists
 * @param carr: The allocated C 2D array to fill
 * @param rsize: Number of rows in the matrix
 * @param csize: Number of columns in the matrix
 * @return Pointer to the filled C 2D array, or NULL on failure
 */
double** conv_pylist_to_carr(PyObject* pylist, double** carr, int rsize, int csize)
{
    int i, j;
    PyObject *row, *item;
    for(i = 0; i < rsize; i++) {
        row = PyList_GetItem(pylist, i);
        if (!PyList_Check(row) || PyList_Size(row) != csize) {
            fprintf(stderr, "An Error Has Occured");
            return NULL;
        }
        for(j = 0; j < csize; j++) {
            item = PyList_GetItem(row, j);
            if (!PyFloat_Check(item)) {
                fprintf(stderr, "An Error Has Occured");
                return NULL;
            }
            carr[i][j] = PyFloat_AsDouble(item);
        }
    }
    return carr;
}


/**
 * Converts a C 2D array (matrix) to a Python list of lists (2D list)
 * @param carr: The input C 2D array
 * @param rsize: Number of rows in the matrix
 * @param csize: Number of columns in the matrix
 * @return Pointer to the created Python list of lists, or NULL on failure
 */
PyObject* conv_carr_to_pylist(double** carr, int rsize, int csize)
{
    int i, j;
    PyObject *pylist = PyList_New(rsize);
    PyObject *row;

    if (pylist == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }
    for(i = 0; i < rsize; i++)
    {
        row = PyList_New(csize);
        for(j = 0; j < csize; j++)
        {
            PyList_SetItem(row, j, PyFloat_FromDouble(carr[i][j]));
        }
        PyList_SetItem(pylist, i, row);
    }
    return pylist;
}


/**
 * Convert a Python list of vectors to a C array
 * This function takes a Python list of vectors (vec_arr_obj) and converts it into a C array (carr).
 * It first parses the Python argument to get the list of vectors, then determines the dimensions
 * of the array (N and vectordim). It allocates memory for the C array and converts the Python list
 * into the C array.
 * 
 * @param args: Python tuple containing a single argument - the Python list of vectors
 * @return Pointer to the allocated and filled C 2D array, or NULL on failure
 */
double** conv_pyvectors_to_carr(PyObject* args)
{
    PyObject* vec_arr_obj;
    double** carr = NULL;

    if (!PyArg_ParseTuple(args, "O", &vec_arr_obj)) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    N = PyList_Size(vec_arr_obj);
    if (N <= 0) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }   

    vectordim = PyList_Size(PyList_GetItem(vec_arr_obj, 0));
    if (vectordim <= 0) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    carr = matrix_malloc(carr, N, vectordim);
    if (carr == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }
    if (conv_pylist_to_carr(vec_arr_obj, carr, N, vectordim) == NULL) {
        matrix_free(carr, N);
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }
    return carr;
}

/**
 * Python wrapper for the matrix_sym function
 * This function serves as a bridge between Python and C, allowing Python code to call the
 * matrix_sym function defined in symnmf.h. It handles the conversion of input arguments
 * from Python to C, calls the C function, and converts the result back to a Python object.
 * 
 * @param args: Python tuple containing a single argument - the Python list of vectors
 * @return Pointer to the resulting Python list of lists (similarity matrix), or NULL on failure
 */
static PyObject* sym_module(PyObject* args)
{
    double** matrix = conv_pyvectors_to_carr(args);
    if (matrix == NULL) {
    fprintf(stderr, "An Error Has Occured");
    return NULL;
    }

    double** sym_matrix = matrix_sym(matrix, N, vectordim);
    matrix_free(matrix, N);
    if (sym_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    PyObject* py_sym_matrix = conv_carr_to_pylist(sym_matrix, N, N);
    matrix_free(sym_matrix, N);
    if (py_sym_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    return Py_BuildValue("O", py_sym_matrix);
}



/**
 * Python wrapper for the matrix_ddg function
 * This function serves as a bridge between Python and C, allowing Python code to call the
 * matrix_ddg function defined in symnmf.h. It handles the conversion of input arguments
 * from Python to C, calls the C function, and converts the result back to a Python object.
 * 
 * @param args: Python tuple containing a single argument - the Python list of vectors
 * @return Pointer to the resulting Python list of lists (diagonal degree matrix), or NULL on failure
 */

static PyObject* ddg_module(PyObject* args)
{
    double** matrix = conv_pyvectors_to_carr(args);
    if (matrix == NULL) {
    fprintf(stderr, "An Error Has Occured");
    return NULL;
    }

    double** ddg_matrix = matrix_ddg(matrix, N, vectordim);
    matrix_free(matrix, N);
    if (ddg_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    PyObject* py_ddg_matrix = conv_carr_to_pylist(ddg_matrix, N, N);
    matrix_free(ddg_matrix, N);
    if (py_ddg_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    return Py_BuildValue("O", py_ddg_matrix);
}

/**
 * Python wrapper for the matrix_norm function
 * This function serves as a bridge between Python and C, allowing Python code to call the
 * matrix_norm function defined in symnmf.h. It handles the conversion of input arguments
 * from Python to C, calls the C function, and converts the result back to a Python object.
 * 
 * @param args: Python tuple containing a single argument - the Python list of vectors
 * @return Pointer to the resulting Python list of lists (normalized similarity matrix), or NULL on failure
 */
static PyObject* norm_module(PyObject* args)
{
    double** matrix = conv_pyvectors_to_carr(args);
    if (matrix == NULL) {
    fprintf(stderr, "An Error Has Occured");
    return NULL;
    }

    double** norm_matrix = matrix_norm(matrix, N, vectordim);
    matrix_free(matrix, N);
    if (norm_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    PyObject* py_norm_matrix = conv_carr_to_pylist(norm_matrix, N, N);
    matrix_free(norm_matrix, N);
    if (py_norm_matrix == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    return Py_BuildValue("O", py_norm_matrix);
}


/**
 * Helper function to convert Python arguments and perform SymNMF computation
 * This function handles the conversion of Python W and H matrices to C arrays,
 * calls the C matrix_symnmf function, and returns the resulting C matrix.
 * This is an internal helper function used by symnmf_module.
 * 
 * @param args: Python tuple containing three arguments - W matrix, H matrix, and integer k
 * @return Pointer to the resulting C 2D array (factor matrix H), or NULL on failure
 */
double** conv_symnmf(PyObject* args)
{
    PyObject* matrix_W_obj;
    PyObject* matrix_H_obj;
    double** matrix_W = NULL;
    double** matrix_H = NULL;

    if (!PyArg_ParseTuple(args, "OOi", &matrix_W_obj, &matrix_H_obj, &k)) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    N = PyList_Size(matrix_W_obj);
    if (N <= 0) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    matrix_W = matrix_malloc(matrix_W, N, N);
    if (matrix_W == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }
    matrix_H = matrix_malloc(matrix_H, N, k);
    if (matrix_H == NULL) {
        matrix_free(matrix_W, N);
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    matrix_W = conv_pylist_to_carr(matrix_W_obj, matrix_W, N, N);
    matrix_H = conv_pylist_to_carr(matrix_H_obj, matrix_H, N, k);

    double** result_H = matrix_symnmf(matrix_W, matrix_H, N, k);
    matrix_free(matrix_W, N);
    matrix_free(matrix_H, N);
    if (result_H == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    return result_H;
}

/**
 * Python wrapper for the SymNMF algorithm
 * This function serves as the main Python interface for SymNMF factorization.
 * It calls the conv_symnmf helper function to perform the computation,
 * then converts the C result back to a Python object for return to Python code.
 * 
 * @param args: Python tuple containing three arguments - W matrix, H matrix, and integer k
 * @return Pointer to the resulting Python list of lists (factor matrix H), or NULL on failure
 */
static PyObject* symnmf_module(PyObject* args)
{
    double** result_H = conv_symnmf(args);
    if (result_H == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    PyObject* py_result_H = conv_carr_to_pylist(result_H, N, k);
    matrix_free(result_H, N);
    if (py_result_H == NULL) {
        fprintf(stderr, "An Error Has Occured");
        return NULL;
    }

    return Py_BuildValue("O", py_result_H);
}


/**
 * Module method definitions
 * This array defines the methods provided by the SymNMF module, including their names,
 * C function pointers, argument types, and documentation strings.
 */
static PyMethodDef SymNMFMethods[] = {
    {"matrix_sym", (PyCFunction)sym_module, METH_VARARGS, "Compute the similarity matrix."},
    {"matrix_ddg", (PyCFunction)ddg_module, METH_VARARGS, "Compute the diagonal degree matrix."},
    {"matrix_norm", (PyCFunction)norm_module, METH_VARARGS, "Compute the normalized similarity matrix."},
    {"symnmf", (PyCFunction)symnmf_module, METH_VARARGS, "Perform SymNMF factorization."},
    {NULL, NULL, 0, NULL} // Sentinel
};


static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "mysymnmfsp",   // name of module
    NULL, // module documentation, may be NULL
    -1,       // size of per-interpreter state of the module,
              // or -1 if the module keeps state in global variables.
    SymNMFMethods
};

PyMODINIT_FUNC PyInit_mysymnmfsp(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (m == NULL)
        return NULL;
    return m;
}

