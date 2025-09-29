#!/bin/bash

echo "=== SymNMF Project Test Suite ==="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_TOTAL=0

# Function to run a test
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_lines="$3"
    
    echo -e "${BLUE}Testing: $test_name${NC}"
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    # Run the command and capture output
    output=$(eval "$command" 2>/dev/null)
    exit_code=$?
    
    if [ $exit_code -eq 0 ] && [ ! -z "$output" ]; then
        # Count lines in output
        line_count=$(echo "$output" | wc -l)
        if [ "$line_count" -eq "$expected_lines" ]; then
            echo -e "${GREEN}✓ PASSED${NC}"
            echo "First few lines of output:"
            echo "$output" | head -3
            echo "..."
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}✗ FAILED - Expected $expected_lines lines, got $line_count${NC}"
            echo "Output preview:"
            echo "$output" | head -5
        fi
    else
        echo -e "${RED}✗ FAILED - Command failed or no output${NC}"
        # Show actual error for debugging
        eval "$command"
    fi
    echo "---"
}

# Function to test compilation
test_compilation() {
    echo -e "${BLUE}Testing: C Program Compilation${NC}"
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    gcc -std=c99 -Wall -Wextra -Werror -pedantic-errors -o symnmf symnmf.c -lm 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED - Compilation successful${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAILED - Compilation failed${NC}"
        echo "Compilation errors:"
        gcc -std=c99 -Wall -Wextra -Werror -pedantic-errors -o symnmf symnmf.c -lm
    fi
    echo "---"
}

# Function to test Python extension compilation
test_python_compilation() {
    echo -e "${BLUE}Testing: Python Extension Compilation${NC}"
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    python3 setup.py build_ext --inplace 2>/dev/null 1>/dev/null
    if [ -f mysymnmfsp.cpython*.so ]; then
        echo -e "${GREEN}✓ PASSED - Python extension compiled${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        
        # Create symlink if needed
        if [ ! -f "mysymnmfsp.so" ]; then
            ln -s mysymnmfsp.cpython*.so mysymnmfsp.so 2>/dev/null
        fi
    else
        echo -e "${RED}✗ FAILED - Python extension compilation failed${NC}"
        python3 setup.py build_ext --inplace
    fi
    echo "---"
}

# Function to test Python import
test_python_import() {
    echo -e "${BLUE}Testing: Python Module Import${NC}"
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    python3 -c "import mysymnmfsp; print('Import successful')" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED - Python module imports correctly${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAILED - Python module import failed${NC}"
        echo "Try creating symlink: ln -s mysymnmfsp.cpython*.so mysymnmfsp.so"
    fi
    echo "---"
}

# Create larger test data (10 points, 2 dimensions) - comma-separated for Python
if [ ! -f "input_1.txt" ]; then
    echo "Creating input_1.txt..."
    cat > input_1.txt << 'EOF'
1.0,1.0
2.0,2.0
3.0,3.0
4.0,4.0
5.0,5.0
6.0,6.0
7.0,7.0
8.0,8.0
9.0,9.0
10.0,10.0
EOF
    echo "✓ input_1.txt created (10x2 data for comma-separated format)"
fi

# Create space-separated version for C program
if [ ! -f "input_1_c.txt" ]; then
    echo "Creating input_1_c.txt for C program..."
    cat > input_1_c.txt << 'EOF'
1.0 1.0
2.0 2.0
3.0 3.0
4.0 4.0
5.0 5.0
6.0 6.0
7.0 7.0
8.0 8.0
9.0 9.0
10.0 10.0
EOF
    echo "✓ input_1_c.txt created (10x2 data for space-separated format)"
fi

# Run compilation tests
test_compilation
if [ -f "setup.py" ]; then
    test_python_compilation
    test_python_import
fi

# Run C program tests (expecting 10x10 output = 10 lines)
if [ -f "./symnmf" ]; then
    echo -e "${BLUE}=== C Program Tests ===${NC}"
    run_test "C Similarity Matrix (sym)" "./symnmf sym input_1_c.txt" 10
    run_test "C Diagonal Degree Matrix (ddg)" "./symnmf ddg input_1_c.txt" 10  
    run_test "C Normalized Matrix (norm)" "./symnmf norm input_1_c.txt" 10
    
    # Test error handling
    echo -e "${BLUE}Testing: C Invalid Arguments${NC}"
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    ./symnmf invalid input_1_c.txt 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${GREEN}✓ PASSED - C correctly handles invalid arguments${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAILED - C should reject invalid arguments${NC}"
    fi
    echo "---"
fi

# Run Python tests (expecting 10x10 output = 10 lines)
if [ -f "mysymnmfsp.so" ] || [ -f mysymnmfsp.cpython*.so ]; then
    echo -e "${BLUE}=== Python Program Tests ===${NC}"
    
    # Test Python script with different goals
    if [ -f "symnmf.py" ]; then
        # Install required packages if missing
        python3 -c "import pandas, numpy" 2>/dev/null || pip3 install pandas numpy 2>/dev/null
        
        run_test "Python SymNMF (sym)" "python3 symnmf.py 3 sym input_1.txt" 10
        run_test "Python SymNMF (ddg)" "python3 symnmf.py 3 ddg input_1.txt" 10
        run_test "Python SymNMF (norm)" "python3 symnmf.py 3 norm input_1.txt" 10
        run_test "Python SymNMF (symnmf)" "python3 symnmf.py 3 symnmf input_1.txt" 10
        
        # Test error handling
        echo -e "${BLUE}Testing: Python Invalid Arguments${NC}"
        TESTS_TOTAL=$((TESTS_TOTAL + 1))
        python3 symnmf.py 3 invalid input_1.txt 2>/dev/null
        if [ $? -ne 0 ]; then
            echo -e "${GREEN}✓ PASSED - Python correctly handles invalid arguments${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}✗ FAILED - Python should reject invalid arguments${NC}"
        fi
        echo "---"
    fi
fi

# Summary
echo -e "${BLUE}=== Test Results ===${NC}"
if [ $TESTS_PASSED -eq $TESTS_TOTAL ]; then
    echo -e "${GREEN}🎉 All tests passed! ($TESTS_PASSED/$TESTS_TOTAL)${NC}"
    echo -e "${GREEN}Your SymNMF implementation is working correctly!${NC}"
else
    echo -e "${RED}Some tests failed. ($TESTS_PASSED/$TESTS_TOTAL passed)${NC}"
    echo -e "${RED}Please check the failed tests above.${NC}"
fi

echo
echo "Test files used:"
echo "- input_1.txt (10x2 comma-separated for Python)"
echo "- input_1_c.txt (10x2 space-separated for C)"
echo "- Expected output: 10x10 matrices (10 lines each)"
echo "- C executable: ./symnmf"
echo "- Python script: symnmf.py"
echo
echo "Cleaning up..."
rm -f symnmf
echo "Done!"