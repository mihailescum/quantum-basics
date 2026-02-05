import pytest

import qiskit as qk

from helper import test_matrix_using_basis, get_matrix_representation

from quantum.gates import ModularInplaceMultiplicationGate
from quantum.utils import reduce_basis_state, combine_basis_state


"""@pytest.mark.parametrize(
    "base, mod, dim_q0, dim_q1",
    [
        (2, 5, 3, 3),
        (2, 5, 3, 5),
        (7, 15, 4, 4),
        (21, 13, 5, 5),
    ],
)
def test_add_swap_basis_states_circuit(base, mod, dim_q0, dim_q1):
    def validation(x):
        q_0, q_1 = reduce_basis_state(x, dim_q0)
        result = combine_basis_state(q_0, q_1 ^ pow(base, q_0, mod), dim_q0)
        return result

    gate = ModularInplaceMultiplicationGate(base, mod, dim_q0, dim_q1)
    native_gate = gate.get_native()
    matrix = get_matrix_representation(native_gate)

    test_matrix_using_basis(matrix, validation)"""
