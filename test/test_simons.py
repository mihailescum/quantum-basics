import pytest

import numpy as np
import numpy.testing as npt

from qiskit import transpile
from qiskit.result import Counts
from qiskit_aer import AerSimulator

from quantum.algorithms import Simons
from quantum.gates import AutoOracleGate


def f(x):
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


def g(x):
    # Function {0,1}^3 \to {0,1}^5 with xor mask s=1
    if x == 0 or x == 3:
        return 19
    elif x == 1 or x == 2:
        return 5
    elif x == 4 or x == 7:
        return 26
    elif x == 5 or x == 6:
        return 1


@pytest.mark.parametrize(
    "counts, expected_result",
    [
        (
            Counts({"1111": 2, "0111": 3, "1000": 1, "1010": 2, "0000": 2, "0101": 3}),
            np.array(
                [
                    [1, 1, 1, 1],
                    [0, 1, 1, 1],
                    [1, 0, 0, 0],
                    [1, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 1],
                ],
            ),
        ),
        (
            Counts({"11111111": 2, "01110000": 3}),
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                ],
            ),
        ),
    ],
)
def test_counts_to_bitmatrix(counts, expected_result):
    result = Simons._counts_to_bitmatrix(counts)
    npt.assert_equal(result, expected_result)


@pytest.mark.parametrize(
    "f, n, m, expected_result",
    [
        (f, 4, 4, 5),
        (lambda x: x, 4, 4, 0),
        (lambda x: x, 2, 2, 0),
        (g, 3, 5, 3),
    ],
)
def test_simons(f, n, m, expected_result):
    def simulate_qc(qc):
        simulator = AerSimulator()
        qct = transpile(qc, backend=simulator)
        return simulator.run(qct, shots=n + 10, memory=False).result().get_counts()

    oracle = AutoOracleGate(f, n, m)
    algorithm = Simons(oracle.gate, n, m)

    result = algorithm.run(simulate_qc)

    assert result == expected_result
