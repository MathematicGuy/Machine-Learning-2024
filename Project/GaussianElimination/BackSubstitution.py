import numpy as np
from RowEchelon import row_echelon
'''
    Gaussian Elimination Module 2: Back Substitution
    Understand Substitution: https://youtu.be/1IHsX1lgpRI?si=8UDUygt71NbI9cB5&t=514
    
    M = [[ 1.   2.   3.   1. ]
         [ 0.   1.   0.   2. ]
         [-0.  -0.   1.   0.8]]
'''


def test_back_substitution(M):
    num_rows = M.shape[0]  # 3
    print(f'Starting Matrix:\n{M}')

    # expect row: 2,1,0
    for row in reversed(range(num_rows)):  # start from 1
        M = M.copy()
        pivot_candidate = M[row, row]
        print(f'CURRENT ROW: {row} | PIVOT: {pivot_candidate}')

        #! Goal: M[row] = M[row] - (value_above_pivot * R[index])
        for index in reversed(range(row)):  # start from 1
            # expect index: 1,0
            print(f'Value above pivot {M[index, row]}')
            #? row_to_reduced = row_to_reduced - value_above_pivot * substitution_row
            print(f'Row to Reduced {index} = {M[index]} - {M[index, row]} * {M[row]}')
            M[index] = M[index] - M[index, row]*M[row]

        print(M)
        print("\n")

    last_column = M[:, -1]
    for i, value in enumerate(last_column):
        print(f'X{i} = {value}')


def back_substitution(M):
    num_rows = M.shape[0]  # 3

    for row in reversed(range(num_rows)):  # start from 1
        M = M.copy()
        # pivot_candidate = M[row, row]

        #! Goal: M[row] = M[row] - (value_above_pivot * R[index])
        for index in reversed(range(row)):  # start from 1
            #? M[index, row] mean value above the pivot with index as row, row as column
            M[index] = M[index] - M[index, row]*M[row]

    # print(f'Back Substitution:\n{M}')
    print("\n")
    return M


def main():
    # A = np.array([[1, 2, 3], [0, 1, 0], [0, 0, 1]])
    # B = np.array([1, 2, 0.8])

    # A = np.array([[1e6, 2e6, 3e6], [4e6, 5e6, 6e6], [7e6, 8e6, 9e6]])
    # B = np.array([6e6, 15e6, 24e6])

    # A = np.array([[2, 4, -1, 5, 2], [1, 2, 0, 3, -1], [0, -2, 0, -2, 3], [0, 0, 1, 2, 4], [0, 0, 0, -3, 6]])
    # B = np.array([2, -1, 3, 4, 6])

    # A = np.array([[0, 5, 0, 2], [1, 2, 3, 6], [0, 2, 7, 1], [3, 22, 1, 0]], dtype=np.dtype(float))
    # B = np.array([[1], [4], [7], [9]])

    '''
    $$
        \left[ \begin{array}{ccc|c}
        1 & -2 & 3 & 7 \\
        0 & 1 & 1 & 4 \\
        0 & 0 & 1 & -10 \\
        \end{array} \right]
    $$
    '''
    A = np.array([[1, -2, 3], [0, 1, 1], [0, 0, 1]])
    B = np.array([7, 4, -10])

    # M = row_echelon(A, B)
    M = np.hstack((A, B.reshape(-1, 1)))  # Augmented matrix creation
    print('Origin:\n', M)
    M = back_substitution(M)
    print(M)

'''
    ?Row Echelon
    [[-0.         -0.         -0.         -0.        ]
     [-0.          1.          2.          3.        ]
     [ 0.33333333  0.66666667  1.          2.        ]]

    ?Back Substitution
    [[-0.         -0.         -0.         -0.        ]
     [-0.          1.          2.          3.        ]
     [ 0.33333333  0.66666667  1.          2.        ]]     
'''

if __name__ == "__main__":
    main()
