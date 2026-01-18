import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum.oracles import AutoOracle
from quantum.utils import combine_basis_state


def f_1_1_const_zero(x: bool) -> int:
    return 0


def f_1_1_const_one(x: bool) -> int:
    return 1


def f_1_1_0_0(x: bool) -> int:
    return 0 if not x else 1


def f_1_1_0_1(x: bool) -> int:
    return 1 if not x else 0


def f_2_1_balanced(x: bool) -> int:
    return 1 if x <= 1 else 0


@pytest.mark.parametrize(
    "f, n, m",
    [
        (f_1_1_const_zero, 1, 1),
        (f_1_1_const_one, 1, 1),
        (f_1_1_0_0, 1, 1),
        (f_1_1_0_1, 1, 1),
        (f_2_1_balanced, 2, 1),
    ],
)
def test_auto_oracle(f, n, m):
    oracle = AutoOracle(f, n, m).gate
    dims = 2 ** (n + m)

    for k in range(0, 2**m):
        for j in range(0, 2**n):
            initial = Statevector.from_int(combine_basis_state(j, k, n), dims)
            expected_result = Statevector.from_int(
                combine_basis_state(j, k ^ f(j), n), dims
            )

            qc = QuantumCircuit(n + m)
            qc.initialize(initial)
            qc.append(oracle, range(n + m))
            result = Statevector(qc)

            assert result.equiv(
                expected_result
            ), f"j={j}, k={k}, expected={expected_result}, result={result}"
