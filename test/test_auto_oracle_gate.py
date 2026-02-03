import pytest

from helper import test_matrix_using_basis, get_matrix_representation

from quantum.gates import AutoOracleGate
from quantum.utils import combine_basis_state, reduce_basis_state


def f_1_1_const_zero(x: int) -> int:
    return 0


def f_1_1_const_one(x: int) -> int:
    return 1


def f_1_1_0_0(x: int) -> int:
    return 0 if not x else 1


def f_1_1_0_1(x: int) -> int:
    return 1 if not x else 0


def f_2_1_balanced(x: int) -> int:
    return 1 if x <= 1 else 0


def f(x: int) -> int:
    # Function {0,1}^4 \to {0,1}^4 with xor mask s=5
    if x == 0 or x == 5:
        return 0
    elif x == 1 or x == 4:
        return 1
    elif x == 2 or x == 7:
        return 2
    elif x == 3 or x == 6:
        return 3
    elif x == 8 or x == 13:
        return 4
    elif x == 9 or x == 12:
        return 5
    elif x == 10 or x == 15:
        return 6
    elif x == 11 or x == 14:
        return 7


def g(x: int) -> int:
    if x == 0 or x == 3:
        return 19
    elif x == 1 or x == 2:
        return 5
    elif x == 4 or x == 7:
        return 26
    elif x == 5 or x == 6:
        return 1


@pytest.mark.parametrize(
    "f, n, m",
    [
        (f_1_1_const_zero, 1, 1),
        (f_1_1_const_one, 1, 1),
        (f_1_1_0_0, 1, 1),
        (f_1_1_0_1, 1, 1),
        (f_2_1_balanced, 2, 1),
        (f, 4, 4),
        (g, 3, 5),
    ],
)
def test_auto_oracle_gate(f, n, m):
    def validation(x):
        q_0, q_1 = reduce_basis_state(x, n)
        r = combine_basis_state(q_0, q_1 ^ f(q_0), n)
        return r

    gate = AutoOracleGate(f, n, m)
    native_gate = gate.get_native()
    matrix = get_matrix_representation(native_gate)

    test_matrix_using_basis(matrix, validation)
