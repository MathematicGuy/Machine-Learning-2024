from RowEchelon import row_echelon
from BackSubstitution import back_substitution
import numpy as np

# ? Step 0: Create Augmented matrix
# Transform matrices A and B into augmented matrix
# Make sure data type is float for accurate calculation

# Test 1: Basic case (already in row-echelon form)
A1 = np.array([[1, 2, -1, 4], [0, 3, 1, 2], [0, 0, 2, -5]])
B1 = np.array([1, 2, 3])

# Test 2: Leading zeros with row swaps
A2 = np.array([[0, 1, 2, 3], [2, 4, 6, 8], [0, 3, 6, 9]])
B2 = np.array([4, 5, 6])

# Test 3: Fractional coefficients
A3 = np.array([[1/2, 1, 3/2], [0, 2/3, 1/3], [0, 0, 5/4]])
B3 = np.array([5/2, 4/3, 7/4])

# Test 4: Zero row
A4 = np.array([[1, -2, 3, 1], [0, 0, 0, 0], [2, -4, 6, 2]])
B4 = np.array([4, 0, 8])

# Test 5: Inconsistent system (no solution)
A5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B5 = np.array([1, 2, 3])

# Test 6: Singular matrix (infinitely many solutions)
A6 = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
B6 = np.array([6, 12, 18])

# Test 7: Large matrix (5x5)
A7 = np.array([[2, 4, -1, 5, 2], [1, 2, 0, 3, -1], [0, -2, 0, -2, 3], [0, 0, 1, 2, 4], [0, 0, 0, -3, 6]])
B7 = np.array([2, -1, 3, 4, 6])

# Test 8: Negative coefficients
A8 = np.array([[-1, 2, -3], [4, -5, 6], [-7, 8, -9]])
B8 = np.array([-4, 5, -6])

# Test 9: Square matrix with determinant zero
A9 = np.array([[2, 4, -2], [4, 8, -4], [6, 12, -6]])
B9 = np.array([2, 4, 6])

# Test 10: Rectangular matrix (more rows than columns)
A10 = np.array([[1, 2], [3, 4], [5, 6]])
B10 = np.array([3, 7, 11])

# Test 11: Rectangular matrix (more columns than rows)
A11 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
B11 = np.array([10, 26])

# Test 12: Identity matrix
A12 = np.eye(3)
B12 = np.array([1, 2, 3])

# Test 13: Matrix with all elements zero except the diagonal
A13 = np.diag([2, 3, 4])
B13 = np.array([2, 3, 4])

# Test 14: Random matrix
A14 = np.random.rand(4, 4)
B14 = np.random.rand(4)

# Test 15: Matrix with large values
A15 = np.array([[1e6, 2e6, 3e6], [4e6, 5e6, 6e6], [7e6, 8e6, 9e6]])
B15 = np.array([6e6, 15e6, 24e6])
#! = None Solution

# Test 16
A16 = np.array([[0, 5, 0, 2], [1, 2, 3, 6], [0, 2, 7, 1], [3, 22, 1, 0]], dtype=np.dtype(float))
B16 = np.array([1, 4, 7, 9])
#! = Error

# Test 17
A17 = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.dtype(float))
B17 = np.array([4, 5, 6])

# Test 18
A18 = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]], dtype=np.dtype(float))
B18 = np.array([7, 8, 9])

# Test 19
A19 = np.array([[3, 6, 9], [12, 15, 18], [21, 24, 27]], dtype=np.dtype(float))
B19 = np.array([10, 11, 12])

# Test 20
A20 = np.array([[4, 8, 12], [16, 20, 24], [28, 32, 36]], dtype=np.dtype(float))
B20 = np.array([13, 14, 15])

'''
[[  3.   6.   6.   8.   1.]
 [  5.   3.   6.   0. -10.]
 [  0.   4.  -5.   8.   8.]
 [  0.   0.   4.   8.   9.]]
'''
A21 = np.array([[3, 6, 6, 8], [5, 3, 6, 0], [0, 4, -5, 8], [0, 0, 4, 8]])
B21 = np.array([1, -10, 8, 9])


def gaussian_elimination(A, B):
    # Perform row echelon form
    M = row_echelon(A, B)
    if isinstance(M, str):
        return M

    print(f'Row Echelon:\n{M}')
    # Perform back substitution (Reversed Row-Echelon)
    M = back_substitution(M)
    print(f'Back Substitution:\n{M}')
    return M


def main():
    # Process each test case
    for i in range(1, 22):
        A = globals()[f"A{i}"]
        B = globals()[f"B{i}"]

        print(f"\nTest {i}:")
        try:
            M = gaussian_elimination(A, B)
            if not isinstance(M, str):
                last_column = M[:, -1]
                print("Solution:")
                for j, value in enumerate(last_column):
                    print(f'X{j} = {value}')

            print(f"Test {i}: Passed")
        except np.linalg.LinAlgError:
            print("System is inconsistent (no solution).")


if __name__ == "__main__":
    main()

