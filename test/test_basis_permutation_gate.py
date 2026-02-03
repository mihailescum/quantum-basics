import pytest

import qiskit as qk

from helper import test_matrix_using_basis, get_matrix_representation

from quantum.gates import BasisPermutationGate


def f(x: int) -> int:
    if x == 0:
        return 7
    elif x == 1:
        return 4
    elif x == 2:
        return 5
    elif x == 3:
        return 2
    elif x == 4:
        return 0
    elif x == 5:
        return 1
    elif x == 6:
        return 6
    elif x == 7:
        return 3


def g(x: int) -> int:
    if x == 0:
        return 1
    elif x == 1:
        return 2
    elif x == 2:
        return 3
    elif x == 3:
        return 4
    elif x == 4:
        return 5
    elif x == 5:
        return 6
    elif x == 6:
        return 7
    elif x == 7:
        return 8
    elif x == 8:
        return 9
    elif x == 9:
        return 10
    elif x == 10:
        return 11
    elif x == 11:
        return 12
    elif x == 12:
        return 13
    elif x == 13:
        return 14
    elif x == 14:
        return 15
    elif x == 15:
        return 0


@pytest.mark.parametrize(
    "a, b, num_qubits",
    [
        (1, 1, 3),
        (0, 7, 3),
    ],
)
def test_add_swap_basis_states_circuit(a, b, num_qubits):
    def validation(x):
        if x == a:
            return b
        elif x == b:
            return a
        else:
            return x

    gate = BasisPermutationGate(lambda x: x, num_qubits)
    qc = qk.QuantumCircuit(num_qubits)
    gate._add_swap_basis_states_circuit(qc, a, b)
    matrix = get_matrix_representation(qc)

    test_matrix_using_basis(matrix, validation)


@pytest.mark.parametrize(
    "f, num_qubits",
    [
        (lambda x: x, 3),
        (f, 3),
        (g, 4),
    ],
)
def test_basis_permutation_gate(f, num_qubits):
    gate = BasisPermutationGate(f, num_qubits)
    native_gate = gate.get_native()
    matrix = get_matrix_representation(native_gate)

    test_matrix_using_basis(matrix, f)
