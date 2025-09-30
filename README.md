# Symmetric Non-negative Matrix Factorization (SymNMF)

This project implements a Symmetric Non-negative Matrix Factorization algorithm in C with Python integration. The implementation includes similarity matrix computation, diagonal degree matrix calculation, and normalized similarity matrix operations.

## Installation

### Prerequisites
- Python 3.x
- C compiler (gcc/clang)
- NumPy
- Pandas

### Build
```bash
make clean
make
```

## Usage

The program can be run with the following command:
```bash
python3 symnmf.py k goal input_file
```

Where:
- `k`: Number of clusters (required for symnmf goal)
- `goal`: One of the following operations:
  - `sym`: Calculate similarity matrix
  - `ddg`: Calculate diagonal degree matrix
  - `norm`: Calculate normalized similarity matrix
  - `symnmf`: Perform Symmetric NMF clustering
- `input_file`: Path to input data file (CSV format)

### Input Format
- CSV file with no headers
- Each row represents a data point
- Values should be numerical

### Output Format
- Matrix output with values formatted to 4 decimal places
- Values are comma-separated

## Implementation Details

### Core Functions
- `matrix_sym`: Computes similarity matrix using Gaussian kernel
- `matrix_ddg`: Computes diagonal degree matrix
- `matrix_norm`: Computes normalized similarity matrix
- `matrix_symnmf`: Performs SymNMF clustering

### Python Integration
The project uses Python C extensions for efficient computation while providing a user-friendly Python interface.

### Memory Management
The implementation includes careful memory management with proper allocation and deallocation of resources.

## Error Handling

The program includes comprehensive error handling for:
- Invalid input parameters
- File operations
- Memory allocation
- Matrix operations

## Project Structure
```
.
├── symnmf.c          # Core C implementation
├── symnmf.h          # Header file with function declarations
├── symnmfmodule.c    # Python C extension implementation
├── symnmf.py         # Python interface
└── Makefile          # Build configuration
```

## Performance Considerations
- Efficient matrix operations
- Optimized memory usage
- Iterative convergence with epsilon threshold

## Example
```bash
python3 symnmf.py 3 symnmf input.txt
```

## License
This project is part of the Software Project course at Tel Aviv University.