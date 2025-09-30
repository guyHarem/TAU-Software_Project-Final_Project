# Symmetric Non-negative Matrix Factorization (SymNMF)

This project implements a Symmetric Non-negative Matrix Factorization algorithm in C with Python integration. The implementation includes certain matrix calculations (similarity, diagonal, normalization) and finally, Symmetric Non-negative Matrix Factorization.
This project also includes an analysis file used to compare the silhouette scores of both SymNMF and KMeans algorithms.

## Installation

### Prerequisites
- Python 3.x
- C compiler (gcc)
- NumPy
- Pandas
- make

### Build
1. Build the extension:
```bash
python3 setup.py build_ext --inplace
```
2. Build the symnmf executable:
```bash
make
```

## Usage

The program can be run in two ways:

### 1. Using Python:
The Python implementation receives 3 arguments:
- `k`: Number of clusters (required for symnmf goal)
- `goal`: One of the following operations:
  - `sym`: Calculate similarity matrix
  - `ddg`: Calculate diagonal degree matrix
  - `norm`: Calculate normalized similarity matrix
  - `symnmf`: Perform Symmetric NMF clustering
- `input_file`: Path to input data file (CSV format)

Example:
```bash
python3 symnmf.py 3 symnmf tests/input_1.txt
```

### 2. Using C:
The C implementation receives 2 arguments:
- `goal`: One of the following operations:
  - `sym`: Calculate similarity matrix
  - `ddg`: Calculate diagonal degree matrix
  - `norm`: Calculate normalized similarity matrix
- `input_file`: Path to input data file (CSV format)

Example:
```bash
./symnmf sym tests/input_1.txt
```

### Comparing Silhouette Scores:
Compare clustering quality between SymNMF and KMeans algorithms:
- `k`: Number of clusters
- `input_file`: Path to input data file (CSV format)

Example:
```bash
python3 analysis.py 3 tests/input_1.txt
```

**Note:** No argument validation is performed! Arguments must be correct and suitable for the required operations.

## Implementation Details

### Core C Functions
- `matrix_sym`: Computes similarity matrix using Gaussian kernel
- `matrix_ddg`: Computes diagonal degree matrix
- `matrix_norm`: Computes normalized similarity matrix
- `matrix_symnmf`: Performs SymNMF clustering

### Python Integration
1. C Extension Module (`symnmfmodule.c`):
   - Implements the interface between Python and C
   - Handles data conversion between Python lists and C arrays
   - Provides memory management for matrix operations

2. Python Interface (`symnmf.py`):
   - Handles command-line argument parsing
   - Reads input data using pandas
   - Initializes matrices for SymNMF using numpy
   - Formats and prints results

## Error Handling
The program includes error handling for:
- File operations
- Memory allocation
- Matrix operations

**Note:** Argument validation is not performed. All arguments should be suitable as mentioned in the Usage section.

## License
This project was written by Guy Harem for the course 'Software Project' at Tel Aviv University.

Course Staff:
- Dr. Mahmood Sharif
- Yael Kupershmidt
- Rami Nasser