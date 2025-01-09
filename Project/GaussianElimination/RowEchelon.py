import array
import numpy as np

'''
    Gaussian Elimination Module 1: Row Echelon
    Step-by-Step Guide:
    Step1: Find the first non-zero element in the first column
        If the first element is 0:
            swap the row with the row having non-zero element
    Step2: Make that element as 1 by dividing the row by that element
    Step3: Make all other elements in the first column as 0 by adding/subtracting the first row
    Step4: Repeat the above steps for all columns  
    
    M = M.copy() -> do this to avoid changing the original Matrix
'''


def augmented_matrix(A, B) -> np.ndarray:
    # Convert matrices into float to prevent integer division
    # B = B.reshape((len(A), 1))  # reshape b to column vector
    Aug_Matrix = np.hstack((A, B.reshape(-1, 1)))  # Augmented matrix creation
    # augMatrix = np.hstack((A, B))
    return Aug_Matrix


def is_square(M) -> bool:
    return M.shape[0] == M.shape[1]


def swap_rows(M: np.ndarray, row_index1: int, row_index2: int) -> np.ndarray:
    M = M.copy()

    '''
        #! If one of the row_index is False, remember:
        #! The False keyword is the same as 0 ( True is the same as 1).

        Matrix slicing: M[[row_index1, row_index2]] mean get row_index1 to row_index2 of M Matrix
        Ex: 
        With row_index1 = 1 and row_index2 = 0 we can swap row_index1 to row_index2 by slicing the matrix  
        M[[2,0]] -> get row 2 to row 0 of M Matrix [[-0. -0. -0. -0.], [ 1.  2.  3.  6.]]
        M[[0,2]] -> get row 0 to row 2 of M Matrix [[ 1.  2.  3.  6.], [-0. -0. -0. -0.]]
    '''
    # print(f'{M[[2, 0]]} \nand\n {M[[0, 2]]}')
    M[[row_index1, row_index2]] = M[[row_index2, row_index1]]

    return M


def get_index_first_non_zero_value_from_column(M, column: int, starting_row: int):
    first_non_zero_index = starting_row
    M = M.copy()

    # Get the first element from starting_row to the last row
    matrix_col = M[starting_row:, column]
    # print('Matrix  col: ', matrix_col)

    #! bug1: what if the last row was 0 -> return the current index position.
    #? Since all value below the pivot is 0. If the current index value is 0, it must be the last row of the matrix
    for i, value in enumerate(matrix_col):  # check each value in the column
        #! bug2: what if the value is close to 0
        #! The Question is what is the Question I want to ask?
        #? How to get the index of the first non-zero value below the current pivot?
        #? The for loop can only repeat 1 time.
        #? How can i add a value from the current index (pivot index) to the first_non_zero_index?
        #? For loop allow me to loop under the condition are met.
        #? Therefor I can get get the total time the loop is repeated and add it to the pivot index
        #? To get the first non-zero value below the pivot.

        if not np.isclose(value, 0, atol=1e-5):  # check if the value is close to 0
            # ? this can only repeat 1. I don't have to if I add first_non_zero_index to index.
            # ? With i as the total repeat time of the for loop.
            first_non_zero_index += i
            return first_non_zero_index

    # return False if there is no non-zero value
    return 0  # 0 mean False


def get_index_first_non_zero_value_from_row(M, row_index, augmented):
    # default for testing
    # print('get_index_first_non_zero_value_from_row')
    M = M.copy()
    # If it is augmented matrix, Ignore the constant values
    if augmented:  # True by default
        M = M[:, :-1]  # take Matrix M from start to the last (:-1) array of the matrix
        print(M)

    row = M[row_index]
    for value in row:
        if value != 0:
            return value


def test_row_echelon(A, B) -> np.ndarray:
    A = A.astype('float64')
    B = B.astype('float64')
    # Transform matrices A and B into augmented matrix
    M = augmented_matrix(A, B)

    # Returns "Singular system" if determinant is zero
    print(A.shape[0], ':', A.shape[1])
    if A.shape[0] != A.shape[1]:
        return 'Singular system'

    # Matrix must be Square to calc Determinant
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):  # True by Default
        print('detA:', det_A)
        return 'Singular system'

    num_rows = len(A)  # M.shape = (4, 3) so M.shape[0] must number of row, M.shape[1] must be number of column

    #! Iterate over the row
    for row in range(num_rows):
        # Echelon row algorithm repeated here 
        pivot_candidate = M[row, row]
        print(f'CURRENT ROW: {row} | PIVOT: {pivot_candidate}')

        #? Swap Row if the pivot candidate is close to 0. atol mean how close it is to zero
        if np.isclose(pivot_candidate, 0, atol=1e-5):
            #? Step 1: Find the first non-zero element in the first column
            first_non_zero_below_pivot_index = get_index_first_non_zero_value_from_column(M, row, row)
            print("Get index first non zero value from column")
            print(f"Column[{row}:{row}]: {first_non_zero_below_pivot_index}")

            #? Step 2: Swap the row with the row having non-zero element
            print(f'Swap row {row} with row {first_non_zero_below_pivot_index}')
            M = swap_rows(M, row, first_non_zero_below_pivot_index)

            # Update the pivot
            pivot = M[row, row]

            #? If the pivot is not zero, then keep the pivot
        else:
            pivot = pivot_candidate

        print(M)

        #! M[index] = M[index] - (1/pivot * M[index-row])
        #? M[1] = M[1] - (1/pivot * M[1-0])
        # Divide the current row by the pivot, so the new pivot will be 1. 
        #? You may use the formula current_row -> 1/pivot * current_row
        # Where current_row can be accessed using M[row].
        M[row] = M[row] * 1 / pivot
        print(f'R{row} after Divided to pivot {pivot}:', M[row])

        print("Row below current row")
        #! M[index, row] -> get the value below the pivot
        for index in range(row + 1, num_rows):  # row + 1 mean the row after the current row
            print(f"Multiplying R{index} with the value {M[index, row]} below the pivot: {pivot}")
            print(f'R{index} -> R{index} - {M[index, row]} * R{row}')
            M[index] = -(M[index] - M[index, row] * M[row])  # minus to make all values positive

        print("\n")
    return M


def row_echelon(A, B) -> np.ndarray:
    A = A.astype('float64')  # A represent coefficient matrix
    B = B.astype('float64')  # B denotes the column vector of constants:
    # Transform matrices A and B into augmented matrix
    M = augmented_matrix(A, B)
    print('Original Matrix: \n', M)

    #? Check if Matrix is Square to calc Determinant then
    if A.shape[0] != A.shape[1]:  # True by default
        return 'Singular system'

    #? Returns "Singular system" if determinant of A is zero
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):  # True by Default
        return 'Singular system'

    num_rows = len(A)  # M.shape = (4, 3) so M.shape[0] must number of row, M.shape[1] must be number of column

    #! Iterate over the row
    for row in range(num_rows):
        # Echelon row algorithm repeated here
        pivot_candidate = M[row, row]

        #? Swap Row if the pivot candidate is close to 0. atol mean how close it is to zero
        if np.isclose(pivot_candidate, 0, atol=1e-5):
            #? Step 1: Find the first non-zero element in the first column
            first_non_zero_below_pivot_index = get_index_first_non_zero_value_from_column(M, row, row)

            #? Step 2: Swap the row with the row having non-zero element
            M = swap_rows(M, row, first_non_zero_below_pivot_index)

            # Update the pivot
            pivot = M[row, row]

            #? If the pivot is not zero, then keep the pivot
        else:
            pivot = pivot_candidate

        #! M[index] = M[index] - (1/pivot * M[index-row])
        #? M[1] = M[1] - (1/pivot * M[1-0])
        # Divide the current row by the pivot, so the new pivot will be 1.
        #? You may use the formula current_row -> 1/pivot * current_row
        # Where current_row can be accessed using M[row].
        M[row] = M[row] * 1 / pivot

        #! M[index, row] -> get the value below the pivot
        for index in range(row + 1, num_rows):  # row + 1 mean the row after the current row
            M[index] = -(M[index] - M[index, row] * M[row])  # minus to make all values positive

    print("\n")
    return M


if __name__ == '__main__':
    #? Step 0: Create Augmented matrix
    # A = np.array([[1, 2, -1, 4], [0, 3, 1, 2], [0, 0, 2, -5]])
    # B = np.array([1, 2, 3])

    '''
        [[1. 2. 3. 1.]
         [0. 1. 0. 2.]
         [0. 0. 5. 4.]]
    '''
    A = np.array([[1, 2, 3], [0, 1, 0], [0, 0, 5]], dtype=np.dtype(float))
    B = np.array([1, 2, 4], dtype=np.dtype(float))

    row_echelon = row_echelon(A, B) # type: ignore
    print(row_echelon)

'''
    [[ 1.          1.          3.         10.        ]
     [ 0.          1.         -0.5         4.5       ]
     [-0.         -0.          1.         -0.27272727]]
'''
