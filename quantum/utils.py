import numpy as np


def combine_basis_state(x, y, dim_x) -> int:
    if x >= (2 << (dim_x - 1)):
        raise ValueError("x has to be smaller than 2 ** dim_x!")

    return y << dim_x | x


def reduce_basis_state(a, dim_first) -> tuple[int, int]:
    x = a & ((2 << (dim_first - 1)) - 1)
    y = a >> dim_first
    return (x, y)


def lower_triangular_form(matrix: np.array) -> np.array:
    for step in range(min(matrix.shape[0], matrix.shape[1])):
        nonzero = np.nonzero(matrix[step:, step])[0]
        if nonzero.size > 0:
            pivot = step + nonzero[0]
            matrix[[step, pivot]] = matrix[[pivot, step]]

            for row in range(step + 1, matrix.shape[0]):
                coefficient = matrix[row, step] / matrix[step, step]
                matrix[row] -= coefficient * matrix[step]
    return matrix
