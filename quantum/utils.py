import string
from qiskit.quantum_info import Statevector

import numpy as np


def combine_basis_state(q_0: int, q_1: int, dim_q0: int) -> int:
    """Given two basis states ´q_0´ and ´q_1´ in spaces of dimension ´2 ** dim_qi´
    respectively, combine them to the basis state ´|x>´ in a space of dimension ´2 ** (dim_q0 + dim_q1)´
    given by |x> = |q_1> ⊗ |q_0>"""

    if q_0 >= (2 << (dim_q0 - 1)):
        raise ValueError("q_0 has to be smaller than 2 ** dim_x!")

    result = q_0 | q_1 << dim_q0
    return result


def reduce_basis_state(basis_state: int, dim_q0: int) -> tuple[int, int]:
    """Given a basis state ´x´ in a space of dimension ´2 ** (dim_q0 + dim_q1)´,
    decompose it into two basis states ´q_0´ and ´q_1´ in space of dimension ´2 ** dim_qi´
    respectively, such that ´|basis_state> = |q_1> ⊗ |q_0>´"""

    q_0 = basis_state & ((2 << (dim_q0 - 1)) - 1)
    q_1 = basis_state >> dim_q0

    return (q_0, q_1)


def split_state(state: string, *num_bits: int):
    state = state[::-1]
    return tuple(
        (
            (
                int(state[sum(num_bits[:i]) : sum(num_bits[: i + 1])][::-1], 2)
                if i + 1 < len(num_bits)
                else int(state[sum(num_bits[:i]) :][::-1], 2)
            )
            for i in range(len(num_bits))
        )
    )


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


def unitary_from_action_on_basis(action, n) -> np.array:
    dims = 2 << (n - 1)
    matrix = np.array(
        [Statevector.from_int(action(basis_state), dims) for basis_state in range(dims)]
    )
    return matrix


def get_bitmask(x, width, big_endian=True):
    x = np.array([np.binary_repr(x, width)], dtype=bytes)
    mask = x.view("S1").reshape((x.size, -1)).astype(int)[0]
    if big_endian:
        return mask[::-1]
    else:
        return mask
